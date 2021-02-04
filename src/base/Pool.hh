//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pool.hh
//---------------------------------------------------------------------------//
#pragma once

#include "PoolTypes.hh"
#include "Types.hh"
#include "detail/PoolImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Reference a contiguous subset of items inside a Pool.
 *
 * \tparam T The value type of items to represent.
 *
 * The template parameter here isn't used directly -- it's more of a marker in
 * a class that contains it. The template parameter must match the
 * corresponding \c Pool type, and more importantly it's only assigned to one
 * particular pool. It doesn't have any persistent connection to its associated
 * pool and thus must be used carefully.
 *
 * The size type is plain "unsigned int" (32-bit in CUDA) rather than
 * \c celeritas::size_type (64-bit) because CUDA currently uses native 32-bit
 * pointer arithmetic. In general this should be the same type as the default
 * OpaqueId::value_type. It's possible that in large problems 4 billion
 * elements won't be enough (for e.g. cross sections), but in that case the
 * PoolBuilder will throw an assertion during construction.
 *
 * \code
 * struct MyMaterial
 * {
 *     real_type number_density;
 *     PoolSlice<ElementComponents> components;
 * };
 *
 * template<Ownership W, MemSpace M>
 * struct MyPools
 * {
 *     Pool<ElementComponents, W, M> components;
 *     Pool<MyMaterial, W, M> materials;
 * };
 * \endcode
 *
 * \todo Not sure if we ever have to directly iterate over the values, but if
 * we wanted to we could have this guy use \c detail::range_iter<unsigned int>
 * instead of unsigned int.
 */
template<class T>
class PoolSlice
{
  public:
    //!@{
    using size_type = detail::pool_size_type;
    //!@}

  public:
    //! Default to an empty slice
    PoolSlice() = default;

    // Construct with a particular range of element indices
    inline CELER_FUNCTION PoolSlice(size_type start, size_type stop);

    //!@{
    //! Range of indices in the corresponding pool
    CELER_CONSTEXPR_FUNCTION size_type start() const { return start_; }
    CELER_CONSTEXPR_FUNCTION size_type stop() const { return stop_; }
    //!@}

    //! Whether the slice is empty
    CELER_CONSTEXPR_FUNCTION bool empty() const { return stop_ == start_; }

    //! Number of elements
    CELER_CONSTEXPR_FUNCTION size_type size() const { return stop_ - start_; }

  private:
    size_type start_{};
    size_type stop_{};
};

//---------------------------------------------------------------------------//
/*!
 * Manage generic array-like data ownership and transfer from host to device.
 *
 * Pools are constructed incrementally on the host, then copied (along with
 * their associated PoolSlice ) to device. A Pool can act as a std::vector<T>,
 * DeviceVector<T>, Span<T>, or Span<const T>. The Spans can point to host or
 * device memory, but the MemSpace template argument protects against
 * accidental accesses from the wrong memory space.
 *
 * Each Pool object is usually accessed with a Slice, which references a
 * contiguous set of elements in the Pool. For example, setup code on the host
 * would extend the Pool with a series of vectors, the addition of which
 * returns a PoolSlice that returns the equivalent data on host or device. This
 * methodology allows complex nested data structures to be built up quickly at
 * setup time without knowing the size requirements beforehand.
 *
 * Host-device functions and classes should use \c Pool with a reference or
 * const_reference Ownership, and the \c MemSpace::native type, which expects
 * device memory when compiled inside a CUDA file and host memory when used
 * inside a C++ source or test. (This design choice prevents a single CUDA file
 * from compiling separate host-compatible and device-compatible compute
 * kernels, but in the case of Celeritas this situation won't arise, because
 * we always want to build host code in C++ files for development ease and to
 * allow testing when CUDA is disabled.)
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
    //// CONSTRUCTION ////

    //!@{
    //! Default constructors
    Pool() = default;
    Pool(const Pool&) = default;
    Pool(Pool&&)      = default;
    //!@}

    // Construct from another pool
    template<Ownership W2, MemSpace M2>
    inline Pool(const Pool<T, W2, M2>& other);

    // Construct from another pool (mutable)
    template<Ownership W2, MemSpace M2>
    inline Pool(Pool<T, W2, M2>& other);

    //!@{
    //! Default assignment
    Pool& operator=(const Pool& other) = default;
    Pool& operator=(Pool&& other) = default;
    //!@}

    // Assign from another pool in the same memory space
    template<Ownership W2>
    inline Pool& operator=(const Pool<T, W2, M>& other);

    // Assign (mutable!) from another pool in the same memory space
    template<Ownership W2>
    inline Pool& operator=(Pool<T, W2, M>& other);

    //// ACCESS ////

    // Access a subset of the data with a slice
    inline CELER_FUNCTION SpanT operator[](const PoolSlice<T>& ps) const;

    // Access a single element
    inline CELER_FUNCTION reference_type operator[](size_type i) const;

    // Direct accesors to underlying data
    CELER_CONSTEXPR_FUNCTION size_type     size() const;
    CELER_CONSTEXPR_FUNCTION bool          empty() const;
    CELER_CONSTEXPR_FUNCTION const_pointer data() const;
    CELER_CONSTEXPR_FUNCTION pointer       data();

  private:
    //// DATA ////

    detail::PoolStorage<T, W, M> storage_{};

    //// FRIENDS ////

    template<class T2, Ownership W2, MemSpace M2>
    friend class Pool;

    template<class T2, MemSpace M2>
    friend class PoolBuilder;

    //!@{
    // Private accessors for pool construction
    using StorageT = typename detail::PoolStorage<T, W, M>::type;
    CELER_FORCEINLINE_FUNCTION const StorageT& storage() const
    {
        return storage_.data;
    }
    CELER_FORCEINLINE_FUNCTION StorageT& storage() { return storage_.data; }
    //@}
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
 * Define an anonymous struct that holds sets of values and references on
 * host and device. This is meant for collections of Pools in a struct
 * that's templated only on the ownership and memory space.
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
