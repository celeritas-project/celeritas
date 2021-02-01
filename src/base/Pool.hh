//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pool.hh
//---------------------------------------------------------------------------//
#pragma once

#include "NumericLimits.hh"
#include "Types.hh"

// Proxy for defining specializations in a separate header that device-only
// code can omit
#define POOL_HOST_HEADER 1
#if POOL_HOST_HEADER
#    include <vector>
#    include "DeviceVector.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Data ownership flag
enum class Ownership
{
    value,    //!< Denotes ownership (possibly shared) of the data
    reference //!< Denotes a reference to the data
};

//! Size/offset type for pool
using pool_size_type = unsigned int;

CELER_CONSTEXPR_FUNCTION size_type max_pool_size()
{
    return static_cast<size_type>(numeric_limits<pool_size_type>::max());
}

//---------------------------------------------------------------------------//
template<class T>
class PoolSpan
{
  public:
    PoolSpan(pool_size_type start, pool_size_type stop)
        : start_(start), stop_(stop)
    {
        CELER_EXPECT(start_ <= stop_);
    }

    //!@{
    //! Range of indices in the corresponding pool
    CELER_CONSTEXPR_FUNCTION pool_size_type start() const { return start_; }
    CELER_CONSTEXPR_FUNCTION pool_size_type stop() const { return stop_; }
    //!@}

    //! Number of elements
    CELER_CONSTEXPR_FUNCTION pool_size_type size() const
    {
        return start_ - stop_;
    }

  private:
    pool_size_type start_{};
    pool_size_type stop_{};
};

//---------------------------------------------------------------------------//
/*!
 * Storage and access to subspans of a contiguous array.
 *
 * Pools are constructed incrementally on the host, then copied (along with
 * their associated PoolSpans) to device.
 */
template<class T, Ownership W, MemSpace M>
class Pool
{
  public:
    Pool& operator=(const Pool& other) = default;

    //! Assign from another pool in the same memory space
    template<class U, Ownership W2>
    Pool& operator=(const Pool<U, W2, M>& other)
    {
        data_ = other.get();
        return *this;
    }

    //! Access all data from this pool
    Span<T> get() const { return data_; }

    //! Access a subspan
    Span<T> operator[](const PoolSpan<T>& ps) const
    {
        CELER_EXPECT(ps.stop() <= data_.size());
        return {data_.data() + ps.start(), data_.data() + ps.size()};
    }

  private:
    Span<T> data_{};
};

#if POOL_HOST_HEADER
template<class T>
class Pool<T, Ownership::value, MemSpace::host>
{
  public:
    using PoolSpanT = PoolSpan<T>;

    Pool() = default;

    // Disallow copy construction and assignment
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    //@{
    //! Access all elements from this pool
    Span<T>       get() { return make_span(data_); }
    Span<const T> get() const { return make_span(data_); }
    //@}

    //! Reserve space for a number of items
    void reserve(size_type count)
    {
        CELER_EXPECT(count <= max_pool_size());
        data_.reserve(count);
    }

    //! Allocate a new number of items
    PoolSpanT allocate(size_type count)
    {
        CELER_EXPECT(count + data_.size() <= max_pool_size());
        pool_size_type start_ = data_.size();
        pool_size_type stop_  = start_ + static_cast<pool_size_type>(count);
        data_.resize(stop_);
        return PoolSpanT(data_, start_, stop_);
    }

  private:
    std::vector<T> data_;
};

template<class T>
class Pool<T, Ownership::value, MemSpace::device>
{
  public:
    Pool() = default;

    //! Construct from host values, copying data directly
    Pool& operator=(const Pool<T, Ownership::value, MemSpace::host>& host_pool)
    {
        Span<const T> host_data = host_pool.get();
        if (!host_data.empty())
        {
            data_.resize(host_data.size());
            data_.copy_to_device(host_data);
        }
        return *this;
    }

    //@{
    //! Access all elements from this pool
    Span<T>       get() { return data_.device_pointers(); }
    Span<const T> get() const { return data_.device_pointers(); }
    //@}

  private:
    DeviceVector<T> data_;
};
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "Pool.i.hh"
