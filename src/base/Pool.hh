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
    value,           //!< Ownership of the data, only on host
    reference,       //!< Mutable reference to the data
    const_reference, //!< Immutable reference to the data
};

//! Size/offset type for pool
using pool_size_type = unsigned int;

CELER_CONSTEXPR_FUNCTION size_type max_pool_size()
{
    return static_cast<size_type>(numeric_limits<pool_size_type>::max());
}

namespace detail
{
template<typename T, Ownership W>
struct ownership_traits
{
    using SpanT          = Span<T>;
    using reference_type = T&;
};

template<typename T>
struct ownership_traits<T, Ownership::const_reference>
{
    using SpanT          = Span<const T>;
    using reference_type = const T&;
};
} // namespace detail

//---------------------------------------------------------------------------//
template<class T>
class PoolRange
{
  public:
    PoolRange(pool_size_type start, pool_size_type stop)
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
 * their associated PoolRanges) to device.
 */
template<class T, Ownership W, MemSpace M>
class Pool
{
  public:
    using SpanT = typename detail::ownership_traits<T, W>::SpanT;
    using reference_type =
        typename detail::ownership_traits<T, W>::reference_type;

    Pool& operator=(const Pool& other) = default;

    //! Assign from another pool in the same memory space
    template<Ownership W2>
    Pool& operator=(const Pool<T, W2, M>& other)
    {
        data_ = other.get();
        return *this;
    }

    //! Access all data from this pool
    SpanT get() const { return data_; }

    //! Access a subspan
    SpanT operator[](const PoolRange<T>& ps) const
    {
        CELER_EXPECT(ps.stop() <= data_.size());
        return {data_.data() + ps.start(), data_.data() + ps.stop()};
    }

    //! Access a single element
    reference_type operator[](size_type idx) const
    {
        CELER_EXPECT(idx < data_.size());
        return data_[idx];
    }

    //! Number of elements
    size_type size() const { return data_.size(); }

  private:
    SpanT data_{};
};

#if POOL_HOST_HEADER
//! The value/host specialization is used to construct and modify.
template<class T>
class Pool<T, Ownership::value, MemSpace::host>
{
  public:
    using SpanT      = Span<T>;
    using PoolRangeT = PoolRange<T>;

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

    //! Number of elements
    size_type size() const { return data_.size(); }

    //! Allocate a new number of items
    PoolRangeT allocate(size_type count)
    {
        CELER_EXPECT(count + data_.size() <= max_pool_size());
        pool_size_type start_ = data_.size();
        pool_size_type stop_  = start_ + static_cast<pool_size_type>(count);
        data_.resize(stop_);
        return PoolRangeT(start_, stop_);
    }

    //! Access a subspan
    SpanT operator[](const PoolRange<T>& ps) const
    {
        CELER_EXPECT(ps.stop() <= data_.size());
        return {data_.data() + ps.start(), data_.data() + ps.stop()};
    }

  private:
    std::vector<T> data_;
};

//! The value/device specialization doesn't need detailed access.
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
            data_ = DeviceVector<T>(host_data.size());
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
