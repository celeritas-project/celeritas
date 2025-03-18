//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/DeviceVector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/cont/InitializedValue.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"

#include "DeviceAllocation.hh"
#include "ObserverPtr.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Host vector for managing uninitialized device-storage data.
 *
 * This is a class used only in host memory (not passed to kernels) to manage
 * device allocation and host/device copies.  It does \em not perform
 * initialization on the data: the host code must define and copy over suitable
 * data.
 *
 * For more complex data usage (dynamic size increases, std vector-like access,
 * object initialization), use \c thrust::device_vector inside a \c .cu file.
 *
 * When a \c StreamId is passed as the last constructor argument,
 * all memory operations are asynchronous and ordered within that stream.
 *
 * \code
    DeviceVector<double> myvec(100);
    myvec.copy_to_device(make_span(hostvec));
    myvec.copy_to_host(make_span(hostvec));
   \endcode
 *
 * - TODO: remove stream? it complicates things
 * - TODO: move to detail since this is basically only a backend for Collection
 */
template<class T>
class DeviceVector
{
#if !CELERITAS_USE_HIP
    // rocrand states have nontrivial destructors, and some HIP integer types
    // are not trivially copyable
    static_assert(std::is_trivially_copyable<T>::value,
                  "DeviceVector element is not trivially copyable");

    static_assert(std::is_trivially_destructible<T>::value,
                  "DeviceVector element is not trivially destructible");
#endif

  public:
    //!@{
    //! \name Type aliases
    using value_type = T;
    using SpanT = Span<T>;
    using SpanConstT = Span<T const>;
    //!@}

  public:
    // Construct with no elements
    DeviceVector() = default;

    // Construct with no elements
    explicit DeviceVector(StreamId stream);

    // Construct with a number of elements
    explicit DeviceVector(size_type count);

    // Construct with a number of elements
    DeviceVector(size_type count, StreamId stream);

    // Swap with another vector
    inline void swap(DeviceVector& other) noexcept;

    // Allocate and copy from host pointers
    void assign(T const* first, T const* last);

    //// ACCESSORS ////

    //! Get the number of elements
    size_type size() const { return size_; }

    //! Whether any elements are stored
    bool empty() const { return size_ == 0; }

    //// DEVICE ACCESSORS ////

    // Copy data to device
    inline void copy_to_device(SpanConstT host_data);

    // Copy data to host
    inline void copy_to_host(SpanT host_data) const;

    // Get a mutable view to device data
    SpanT device_ref() { return {this->data(), this->size()}; }

    // Get a const view to device data
    SpanConstT device_ref() const { return {this->data(), this->size()}; }

    // Raw pointer to device data (dangerous!)
    inline T* data();

    // Raw pointer to device data (dangerous!)
    inline T const* data() const;

  private:
    DeviceAllocation allocation_;
    InitializedValue<size_type> size_;
};

// Swap two vectors.
template<class T>
inline void swap(DeviceVector<T>& a, DeviceVector<T>& b) noexcept;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a stream.
 */
template<class T>
DeviceVector<T>::DeviceVector(StreamId stream) : allocation_{stream}, size_{0}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a number of allocated elements.
 */
template<class T>
DeviceVector<T>::DeviceVector(size_type count)
    : allocation_{count * sizeof(T)}, size_{count}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a number of allocated elements and a stream.
 *
 * To make resizing eaasier, the stream may be null.
 */
template<class T>
DeviceVector<T>::DeviceVector(size_type count, StreamId stream)
{
    if (stream)
    {
        allocation_ = DeviceAllocation{count * sizeof(T), stream};
    }
    else
    {
        allocation_ = DeviceAllocation{count * sizeof(T)};
    }
    size_ = count;
}

//---------------------------------------------------------------------------//
/*!
 * Get the device data pointer.
 */
template<class T>
void DeviceVector<T>::swap(DeviceVector& other) noexcept
{
    using std::swap;
    swap(size_, other.size_);
    swap(allocation_, other.allocation_);
}

//---------------------------------------------------------------------------//
/*!
 * Allocate and copy from \em host pointers.
 *
 * Not exception safe: if the copy fails, the original contents are lost.
 */
template<class T>
void DeviceVector<T>::assign(T const* first, T const* last)
{
    auto const new_size = static_cast<size_type>(last - first);
    if (new_size > size_ && new_size * sizeof(T) > allocation_.size())
    {
        // Reallocate
        *this = DeviceVector<T>(new_size, allocation_.stream_id());
    }
    else
    {
        // Update size to fit capacity
        size_ = new_size;
    }

    this->copy_to_device({first, new_size});
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to device.
 */
template<class T>
void DeviceVector<T>::copy_to_device(SpanConstT data)
{
    CELER_EXPECT(data.size() == this->size());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    allocation_.copy_to_device({reinterpret_cast<std::byte const*>(data.data()),
                                data.size() * sizeof(T)});
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to host.
 */
template<class T>
void DeviceVector<T>::copy_to_host(SpanT data) const
{
    CELER_EXPECT(data.size() == this->size());
    allocation_.copy_to_host(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        {reinterpret_cast<std::byte*>(data.data()), data.size() * sizeof(T)});
}

//---------------------------------------------------------------------------//
/*!
 * Get a device data pointer.
 */
template<class T>
T* DeviceVector<T>::data()
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<T*>(allocation_.device_ref().data());
}

//---------------------------------------------------------------------------//
/*!
 * Get a device data pointer.
 */
template<class T>
T const* DeviceVector<T>::data() const
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<T const*>(allocation_.device_ref().data());
}

//---------------------------------------------------------------------------//
/*!
 * Swap two vectors.
 */
template<class T>
void swap(DeviceVector<T>& a, DeviceVector<T>& b) noexcept
{
    return a.swap(b);
}

//---------------------------------------------------------------------------//
/*!
 * Prevent accidental construction of Span from a device vector.
 *
 * Use \c dv.device_ref() to get a span.
 */
template<class T>
CELER_FUNCTION Span<T const> make_span(DeviceVector<T> const& dv)
{
    static_assert(sizeof(T) == 0, "Cannot 'make_span' from a device vector");
    return {dv.data(), dv.size()};
}

//---------------------------------------------------------------------------//
//! Prevent accidental construction of Span from a device vector.
template<class T>
CELER_FUNCTION Span<T> make_span(DeviceVector<T>& dv)
{
    static_assert(sizeof(T) == 0, "Cannot 'make_span' from a device vector");
    return {dv.data(), dv.size()};
}

//---------------------------------------------------------------------------//
//! Create an observer pointer from a device vector.
template<class T>
ObserverPtr<T, MemSpace::device> make_observer(DeviceVector<T>& vec) noexcept
{
    return ObserverPtr<T, MemSpace::device>{vec.data()};
}

//---------------------------------------------------------------------------//
//! Create an observer pointer from a pointer in the native memspace.
template<class T>
ObserverPtr<T const, MemSpace::device>
make_observer(DeviceVector<T> const& vec) noexcept
{
    return ObserverPtr<T const, MemSpace::device>{vec.data()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
