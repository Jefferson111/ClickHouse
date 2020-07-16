#pragma once

#include <string.h>

#include <sse2.h>
#include <avx512f.h>


/** memcpy function could work suboptimal if all the following conditions are met:
  * 1. Size of memory region is relatively small (approximately, under 50 bytes).
  * 2. Size of memory region is not known at compile-time.
  *
  * In that case, memcpy works suboptimal by following reasons:
  * 1. Function is not inlined.
  * 2. Much time/instructions are spend to process "tails" of data.
  *
  * There are cases when function could be implemented in more optimal way, with help of some assumptions.
  * One of that assumptions - ability to read and write some number of bytes after end of passed memory regions.
  * Under that assumption, it is possible not to implement difficult code to process tails of data and do copy always by big chunks.
  *
  * This case is typical, for example, when many small pieces of data are gathered to single contiguous piece of memory in a loop.
  * - because each next copy will overwrite excessive data after previous copy.
  *
  * Assumption that size of memory region is small enough allows us to not unroll the loop.
  * This is slower, when size of memory is actually big.
  *
  * Use with caution.
  */

namespace detail
{
    inline void memcpySmallAllowReadWriteOverflow15Impl(char * __restrict dst, const char * __restrict src, ssize_t n)
    {
        while (n > 0)
        {
            simde_mm_storeu_si128(reinterpret_cast<simde__m128i *>(dst),
                simde_mm_loadu_si128(reinterpret_cast<const simde__m128i *>(src)));

            dst += 16;
            src += 16;
            n -= 16;
        }
    }

    inline void memcpySmallAllowReadWriteOverflow63Impl(char * __restrict dst, const char * __restrict src, ssize_t n)
    {
        while (n > 0)
        {
            simde_mm512_storeu_si512(reinterpret_cast<simde__m512i *>(dst),
                                     simde_mm512_loadu_si512(reinterpret_cast<const simde__m512i *>(src)));

            dst += 64;
            src += 64;
            n -= 64;
        }
    }
}

/** Works under assumption, that it's possible to read up to 15 excessive bytes after end of 'src' region
  *  and to write any garbage into up to 15 bytes after end of 'dst' region.
  */
inline void memcpySmallAllowReadWriteOverflow15(void * __restrict dst, const void * __restrict src, size_t n)
{
    detail::memcpySmallAllowReadWriteOverflow15Impl(reinterpret_cast<char *>(dst), reinterpret_cast<const char *>(src), n);
}

/** NOTE There was also a function, that assumes, that you could read any bytes inside same memory page of src.
  * This function was unused, and also it requires special handling for Valgrind and ASan.
  */

inline void memcpySmallAllowReadWriteOverflow63(void * __restrict dst, const void * __restrict src, size_t n)
{
    detail::memcpySmallAllowReadWriteOverflow63Impl(reinterpret_cast<char *>(dst), reinterpret_cast<const char *>(src), n);
}
