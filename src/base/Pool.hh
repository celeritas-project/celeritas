//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pool.hh
//---------------------------------------------------------------------------//
#pragma once

#include "PoolTypes.hh"
#include "Span.hh"
#include "Types.hh"
#include "detail/PoolImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class T>
class PoolRange
{
  public:
    //!@{
    using size_type = unsigned int;
    //!@}

  public:
    //! Default to an empty range
    PoolRange() = default;

    //! Construct with a particular range
    PoolRange(size_type start, size_type stop) : start_(start), stop_(stop)
    {
        CELER_EXPECT(start_ <= stop_);
    }

    //!@{
    //! Range of indices in the corresponding pool
    CELER_CONSTEXPR_FUNCTION size_type start() const { return start_; }
    CELER_CONSTEXPR_FUNCTION size_type stop() const { return stop_; }
    //!@}

    //! Number of elements
    CELER_CONSTEXPR_FUNCTION size_type size() const { return stop_ - start_; }

  private:
    size_type start_{};
    size_type stop_{};
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
    Pool(const Pool&) = default;
    Pool(Pool&&)      = default;

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
    CELER_FUNCTION SpanT operator[](const PoolRange<T>& ps) const
    {
        CELER_EXPECT(ps.stop() <= this->size());
        return {this->data() + ps.start(), this->data() + ps.stop()};
    }

    //! Access a single element
    CELER_FUNCTION reference_type operator[](size_type idx) const
    {
        return d_.data[idx];
    }

    //! Access a subspan
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

    //!@{
    // Private accessors for pool construction
    using StorageT = typename detail::PoolStorage<T, W, M>::type;
    const StorageT& storage() const { return d_.data; }
    StorageT&       storage() { return d_.data; }
    //@}

    template<class T2, MemSpace M2>
    friend class PoolBuilder;
};

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
