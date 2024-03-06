#include <assert.h>
#include <stdint.h>

#include <unistd.h>
#include <sys/mman.h>

template<typename T>
constexpr T Pow2Align(T value, uint64_t alignment) {
  // assert(IsPowerOfTwo(alignment));
  return (
    (value + static_cast<T>(alignment) - 1) & ~(static_cast<T>(alignment) - 1)
  );
}

constexpr void const* VoidPtrInc(void const* p, size_t numBytes) {
  return (static_cast<uint8_t const*>(p) + numBytes);
}

constexpr void* VoidPtrInc(void* p, size_t numBytes) {
  return (static_cast<uint8_t*>(p) + numBytes);
}

inline void* VoidPtrAlign(void* ptr, size_t alignment) {
  // assert(IsPowerOfTwo(alignment));

  return reinterpret_cast<void*>(
    (reinterpret_cast<size_t>(ptr) + (alignment - 1)) & ~(alignment - 1)
  );
}

constexpr size_t VoidPtrDiff(void const* p1, void const* p2) {
  assert(p1 >= p2);
  return (static_cast<uint8_t const*>(p1) - static_cast<uint8_t const*>(p2));
}

size_t VirtualPageSize() {
  return sysconf(_SC_PAGESIZE);
}

bool VirtualReserve(
  size_t sizeInBytes, void** ppOut, void* pMem, size_t alignment
) {
  bool result = true;

  if(sizeInBytes == 0) {
    result = false;
  } else if(ppOut == nullptr) {
    result = false;
  }

  if(result == true) {
    void* pMemory =
      mmap(pMem, sizeInBytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if((pMemory != nullptr) && (pMemory != MAP_FAILED)) {
      assert(ppOut != nullptr);
      (*ppOut) = pMemory;
    } else {
      result = false;
    }
  }

  return result;
}

bool VirtualCommit(void* pMem, size_t sizeInBytes, bool isExecutable) {
  bool result = true;

  if(sizeInBytes == 0) {
    result = false;
  } else if(pMem == nullptr) {
    result = false;
  }

  if(result == true) {
    int32_t protFlags = PROT_READ | PROT_WRITE;
    protFlags         = isExecutable ? (protFlags | PROT_EXEC) : protFlags;

    void* pMemory = mmap(
      pMem,
      sizeInBytes,
      protFlags,
      MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0
    );

    if((pMemory != pMem) || (pMemory == MAP_FAILED)) {
      result = false;
    }
  }

  return result;
}

bool VirtualDecommit(void* pMem, size_t sizeInBytes) {
  bool result = true;

  if(sizeInBytes == 0) {
    result = false;
  } else if(pMem == nullptr) {
    result = false;
  }

  if(result == true) {
    void* pMemory = mmap(
      pMem,
      sizeInBytes,
      PROT_NONE,
      MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0
    );

    if((pMemory != pMem) || (pMemory == MAP_FAILED)) {
      result = false;
    }
  }

  return result;
}

bool VirtualRelease(void* pMem, size_t sizeInBytes) {
  bool result = true;

  if(sizeInBytes == 0) {
    result = false;
  } else if(pMem == nullptr) {
    result = false;
  }

  if(result == true) {
    result = false;

    int releaseResult = munmap(pMem, sizeInBytes);

    if(releaseResult == 0) {
      result = true;
    }
  }

  return result;
}

struct AllocInfo {
    AllocInfo(size_t bytes, size_t alignment, bool zeroMem)
      : bytes(bytes)
      , alignment(alignment)
      , zeroMem(zeroMem) {
    }

    size_t       bytes;
    size_t const alignment;
    bool const   zeroMem;
};

struct FreeInfo {
    FreeInfo(void* pClientMem)
      : pClientMem(pClientMem) {
    }

    void* pClientMem;
};

namespace Util {
  class VirtualLinearAllocator {
    public:
      VirtualLinearAllocator(size_t size)
        : m_pStart(nullptr)
        , m_pCurrent(nullptr)
        , m_size(size)
        , m_pageSize(0) {
      }

      virtual ~VirtualLinearAllocator() {
        if(m_pStart != nullptr) {
          bool result = VirtualRelease(m_pStart, m_size);
          assert(result == true);
        }
      }

      bool Init() {
        m_pageSize = VirtualPageSize();
        m_size     = Pow2Align(m_size, m_pageSize);

        bool result = VirtualReserve(m_size, &m_pStart, nullptr, 1);

        if(result == true) {
          result = VirtualCommit(m_pStart, m_pageSize, false);
        }

        if(result == true) {
          m_pCurrent         = m_pStart;
          m_pCommittedToPage = VoidPtrInc(m_pCurrent, m_pageSize);
        }

        return result;
      }

      void* Alloc(AllocInfo const& allocInfo) {
        void* pAlignedCurrent = VoidPtrAlign(m_pCurrent, allocInfo.alignment);
        void* pNextCurrent    = VoidPtrInc(pAlignedCurrent, allocInfo.bytes);
        void* pAlignedEnd     = VoidPtrAlign(pNextCurrent, m_pageSize);

        if(allocInfo.bytes > Remaining()) {
          pAlignedCurrent = nullptr;
        } else if(pAlignedEnd > m_pCommittedToPage) {
          size_t const commitBytes =
            VoidPtrDiff(pAlignedEnd, m_pCommittedToPage);

          bool const result =
            VirtualCommit(m_pCommittedToPage, commitBytes, false);

          if(result == true) {
            m_pCommittedToPage = VoidPtrInc(m_pCommittedToPage, commitBytes);
            m_pCurrent         = pNextCurrent;
          } else {
            pAlignedCurrent = nullptr;
          }
        } else {
          m_pCurrent = pNextCurrent;
        }

        return pAlignedCurrent;
      }

      void Free(FreeInfo const& freeInfo) {
      }

      void Rewind(void* pStart, bool decommit) {
        assert((m_pStart <= pStart) && (pStart <= m_pCurrent));

        if(pStart != m_pCurrent) {
          if(decommit) {
            void* pStartPage = VoidPtrAlign(VoidPtrInc(pStart, 1), m_pageSize);
            void* pCurrentPage = VoidPtrAlign(m_pCurrent, m_pageSize);
            size_t const numPages =
              VoidPtrDiff(pCurrentPage, pStartPage) / m_pageSize;

            if(numPages > 0) {
              bool result = VirtualDecommit(pStartPage, m_pageSize * numPages);
              assert(result == true);

              m_pCommittedToPage = pStartPage;
            }
          }

          m_pCurrent = pStart;
        }
      }

      void* Current() {
        return m_pCurrent;
      }

      void* Start() {
        return m_pStart;
      }

      size_t BytesAllocated() {
        return VoidPtrDiff(m_pCurrent, m_pStart);
      }

      size_t Remaining() const {
        return m_size - VoidPtrDiff(m_pCurrent, m_pStart);
      }

    private:
      void* m_pStart;
      void* m_pCurrent;
      void* m_pCommittedToPage;

      size_t m_size;
      size_t m_pageSize;
  };

  template<class LinearAllocator>
  class LinearAllocatorAuto {
    public:
      LinearAllocatorAuto(LinearAllocator* pAllocator, bool decommit)
        : m_pAllocator(pAllocator)
        , m_pStart(nullptr)
        , m_decommit(decommit) {
        assert(pAllocator != nullptr);
        m_pStart = m_pAllocator->Current();
      }

      ~LinearAllocatorAuto() {
        m_pAllocator->Rewind(m_pStart, m_decommit);
      }

      void* Alloc(AllocInfo const& allocInfo) {
        void* pMemory = m_pAllocator->Alloc(allocInfo);

        return pMemory;
      }

      void Free(FreeInfo const& freeInfo) {
        m_pAllocator->Free(freeInfo);
      }

    private:
      LinearAllocator* const m_pAllocator;

      void*      m_pStart;
      bool const m_decommit;
  };

} // namespace Util
