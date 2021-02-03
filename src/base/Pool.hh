//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pool.hh
//---------------------------------------------------------------------------//
#pragma once

#include "NumericLimits.hh"
#include "Span.hh"
#include "PoolTypes.hh"
#include "detail/PoolImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

CELER_CONSTEXPR_FUNCTION size_type max_pool_size()
{
    return static_cast<size_type>(numeric_limits<pool_size_type>::max());
}

//---------------------------------------------------------------------------//
template<class T>
class PoolRange
{
  public:
    //! Default to an empty range
    PoolRange() = default;

    //! Construct with a particular range
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
    using PoolTraitsT = detail::PoolTraits<T, W>;

  public:
    //!@{
    //! Type aliases
    using SpanT                = typename PoolTraitsT::SpanT;
    using SpanConstT           = typename PoolTraitsT::SpanConstT;
    using value_type           = typename PoolTraitsT::value_type;
    using pointer              = typename PoolTraitsT::pointer;
    using const_pointer        = typename PoolTraitsT::const_pointer;
    using reference_type       = typename PoolTraitsT::reference_type;
    using const_reference_type = typename PoolTraitsT::const_reference_type;
    //!@}

  public:
    //! Default constructor
    Pool() = default;

    //! Construct from another pool
    template<Ownership W2, MemSpace M2>
    Pool(const Pool<T, W2, M2>& other)
        : d_(detail::PoolAssigner<W, M>()(other.d_))
    {
    }

    //! Construct from another pool (mutable)
    template<Ownership W2, MemSpace M2>
    Pool(Pool<T, W2, M2>& other) : d_(detail::PoolAssigner<W, M>()(other.d_))
    {
    }

    // Default assignment
    Pool& operator=(const Pool& other) = default;
    Pool& operator=(Pool&& other) = default;

    //! Assign from another pool in the same memory space
    template<Ownership W2>
    Pool& operator=(const Pool<T, W2, M>& other)
    {
        d_ = detail::PoolAssigner<W, M>()(other.d_);
        return *this;
    }

    //! Assign (mutable!) from another pool in the same memory space
    template<Ownership W2>
    Pool& operator=(Pool<T, W2, M>& other)
    {
        d_ = detail::PoolAssigner<W, M>()(other.d_);
        return *this;
    }

    //! Access a subspan
    CELER_FUNCTION SpanT operator[](const PoolRange<T>& ps)
    {
        CELER_EXPECT(ps.stop() <= this->size());
        return {this->data() + ps.start(), this->data() + ps.stop()};
    }

    //! Access a single element
    CELER_FUNCTION reference_type operator[](size_type idx)
    {
        CELER_EXPECT(idx < this->size());
        return d_.data[idx];
    }

    //! Access a subspan
    CELER_FUNCTION SpanConstT operator[](const PoolRange<T>& ps) const
    {
        CELER_EXPECT(ps.stop() <= this->size());
        return {this->data() + ps.start(), this->data() + ps.stop()};
    }

    //! Access a single element
    CELER_FUNCTION const_reference_type operator[](size_type idx) const
    {
        CELER_EXPECT(idx < this->size());
        return d_.data[idx];
    }

    //!@{
    //! Forward to local data class
    CELER_FORCEINLINE_FUNCTION size_type size() const
    {
        return d_.data.size();
    }
    CELER_FORCEINLINE_FUNCTION bool empty() const { return d_.data.empty(); }
    CELER_FORCEINLINE_FUNCTION const_pointer data() const
    {
        return d_.data.data();
    }
    CELER_FORCEINLINE_FUNCTION pointer data() { return d_.data.data(); }
    //!@}

  private:
    detail::PoolStorage<T, W, M> d_{};

    template<class T2, Ownership W2, MemSpace M2>
    friend class Pool;
};

#if 0
template<class T>
class PoolBuilder
{
};
    //! Reserve space for a number of items
    void reserve(size_type count)
    {
        CELER_EXPECT(count <= max_pool_size());
        data_.reserve(count);
    }

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
    SpanT operator[](const PoolRange<T>& ps)
    {
        CELER_EXPECT(ps.stop() <= data_.size());
        return {data_.data() + ps.start(), data_.data() + ps.stop()};
    }

    //! Access a subspan
    Span<const T> operator[](const PoolRange<T>& ps) const
    {
        CELER_EXPECT(ps.stop() <= data_.size());
        return {data_.data() + ps.start(), data_.data() + ps.stop()};
    }

    //!@{
    //! Forward to storage
    CELER_FUNCTION size_type     size() const { return data_.size(); }
    CELER_FUNCTION bool          empty() const { return data_.empty(); }
    CELER_FUNCTION const_pointer data() const { return data_.data(); }
    CELER_FUNCTION pointer       data() { return data_.data(); }
    //!@}

  private:
    std::vector<T> data_;
};

//! The value/device specialization doesn't need detailed access.
template<class T>
class Pool<T, Ownership::value, MemSpace::device>
{
  public:
    using const_pointer = const T*;
    using pointer       = T*;
    using value_type    = T;

    Pool() = default;

    //! Construct with a specific number of elements
    explicit Pool(size_type size) : data_(size) {}

    //! Construct from host values, copying data directly
    Pool& operator=(const Pool<T, Ownership::value, MemSpace::host>& host_pool)
    {
        Span<const T> host_data = make_span(host_pool);
        if (!host_data.empty())
        {
            data_ = DeviceVector<T>(host_data.size());
            data_.copy_to_device(host_data);
        }
        return *this;
    }

    //! Resize (TODO: rethink this: whether to add resizing to DeviceVector?)
    void resize(size_type size)
    {
        CELER_EXPECT(data_.empty() || size <= data_.capacity());
        if (data_.empty())
        {
            data_ = DeviceVector<T>(size);
        }
        else
        {
            data_.resize(size);
        }
    }

    //!@{
    //! Forward to storage
    CELER_FUNCTION size_type size() const { return data_.size(); }
    CELER_FUNCTION bool      empty() const { return data_.empty(); }
    CELER_FUNCTION const_pointer data() const
    {
        return data_.device_pointers().data();
    }
    CELER_FUNCTION pointer data() { return data_.device_pointers().data(); }
    //!@}

  private:
    DeviceVector<T> data_;
};
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
/*!
 * \def CELER_POOL_TYPE(CLSNAME, OWNERSHIP, MEMSPACE)
 *
 * Define type alias for a template class (usually a collection of pools) with
 * the given ownership and memory space.
 */
#define CELER_POOL_TYPE(CLSNAME, OWNERSHIP, MEMSPACE) \
    CLSNAME<::celeritas::Ownership::OWNERSHIP, ::celeritas::MemSpace::MEMSPACE>

//---------------------------------------------------------------------------//
/*!
 * \def CELER_POOL_STRUCT
 *
 * Define an anonymous struct that holds sets of value/reference for
 * host/device.
 *
 * Example:
 * \code
 * class FooParams
 * {
 *  public:
 *   using PoolDeviceRef = CELER_POOL_TYPE(FooPools, const_reference, device);
 *
 *   const PoolDeviceRef& device_pointers() const {
 *    return pools_.device_ref;
 *   }
 *  private:
 *   CELER_POOL_STRUCT(FooPools, const_reference) pools_;
 * };
 * \endcode
 */
#define CELER_POOL_STRUCT(CLSNAME, REFTYPE)                   \
    struct                                                    \
    {                                                         \
        CELER_POOL_TYPE(CLSNAME, value, host) host;           \
        CELER_POOL_TYPE(CLSNAME, value, device) device;       \
        CELER_POOL_TYPE(CLSNAME, REFTYPE, host) host_ref;     \
        CELER_POOL_TYPE(CLSNAME, REFTYPE, device) device_ref; \
    }

//---------------------------------------------------------------------------//

#include "Pool.i.hh"
