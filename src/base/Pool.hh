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
#include <memory>
#include <vector>
#include "DeviceVector.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Data ownership flag
enum class Ownership
{
    value, //!< Denotes ownership (possibly shared) of the data
    reference //!< Denotes a reference to the data
};

// Forward declare the Pool class
template<class T, Ownership W, MemSpace M> class Pool;

//! Size/offset type for pool
using pool_size_type = unsigned int;

CELER_CONSTEXPR_FUNCTION size_type max_pool_size()
{
    return static_cast<size_type>(numeric_limits<pool_size_type>::max());
}

//---------------------------------------------------------------------------//
/*!
 *
 */
template<class T, Ownership W, MemSpace M>
class PoolItem
{
  public:
    using PoolT = Pool<T, Ownership::value, MemSpace::host>;

    PoolItem& operator=(const PoolItem& other) = default;

    //! Assign from another pool
    template<Ownership W2, MemSpace M2>
    PoolItem& operator=(const PoolItem<T, W2, M2>& other)
    {
        start_ = other.start();
        stop_ = other.stop();
        return *this;
    }

    //! Access data from this pool item
    inline Span<const T> get(const PoolT& pool) const;

    //!@{
    //! Range of indices in the corresponding pool
    CELER_FUNCTION pool_size_type start() const { return start_; }
    CELER_FUNCTION pool_size_type stop() const { return stop_; }
    //!@}

  private:
    pool_size_type start_{};
    pool_size_type stop_{};
};

// Declare host owned (construction-friendly) specializations
template<class T>
class PoolItem<T, Ownership::value, MemSpace::host>;

#if POOL_HOST_HEADER

template<class T>
class PoolItem<T, Ownership::value, MemSpace::host>
{
  public:
    using PoolT = Pool<T, Ownership::value, MemSpace::host>;

    // Construct from a pool, checking for size limitations.
    PoolItem(std::weak_ptr<const std::vector<T>> data, pool_size_type start, pool_size_type stop)
        : data_(std::move(data))
        , start_(start)
        , stop_(stop)
    {
#if CELERITAS_DEBUG
        auto shared_data = data_.lock();
        CELER_EXPECT(shared_data);
        CELER_EXPECT(start_ <= stop_);
        CELER_EXPECT(stop_ <= shared_data->size());
#endif
    }

    PoolItem& operator=(const PoolItem& other) = default;

    //! Access data from this pool item
    inline Span<const T> get(const PoolT& pool) const;

    //! Access data from this pool item using the shared pointer
    Span<T> get()
    {
        auto shared_data = data_.lock();
        CELER_EXPECT(shared_data);
        CELER_EXPECT(shared_data->size() >= static_cast<size_type>(stop_));
        T* begin_pool = shared_data->data();
        return {begin_pool + start_, begin_pool + stop_};
    }

    //!@{
    //! Range of indices in the corresponding pool
    CELER_FUNCTION pool_size_type start() const { return start_; }
    CELER_FUNCTION pool_size_type stop() const { return stop_; }
    //!@}

  private:
    std::weak_ptr<const std::vector<T>> data_;
    pool_size_type start_{};
    pool_size_type stop_{};
};

#endif

//---------------------------------------------------------------------------//
/*!
 * Storage and access to subspans of a contiguous array.
 *
 * Pools are constructed incrementally on the host, then copied (along with
 * their associated PoolItems) to device.
 */
template<class T, Ownership W, MemSpace M>
class Pool
{
  public:
    Pool& operator=(const Pool& other) = default;

    //! Assign from another pool in the same memory space
    template<Ownership W2>
    Pool& operator=(const Pool<T, W2, M>& other)
    {
        data_ = other.get();
        return *this;
    }

    //! Access all elements from this pool
    Span<const T> get() const { return data_; }

  private:
    Span<T> data_{};
};

// Declare specializations with ownership of data
template<class T>
class PoolItem<T, Ownership::value, MemSpace::host>;
template<class T>
class PoolItem<T, Ownership::value, MemSpace::device>;

#if POOL_HOST_HEADER
template<class T>
class Pool<T, Ownership::value, MemSpace::host>
{
  public:
    using PoolItemT = PoolItem<T, Ownership::value, MemSpace::host>;

    inline Pool();

    // Disallow copy construction and assignment
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    //! Access all elements from this pool
    Span<const T> get() const { return make_span(*data_); }

    //! Reserve space for a number of items
    void reserve(size_type count)
    {
        CELER_EXPECT(count <= max_pool_size());
        data_->reserve(count);
    }

    //! Allocate a new number of items
    PoolItemT allocate(size_type count)
    {
        CELER_EXPECT(count + data_->size() <= max_pool_size());
        pool_size_type start_ = data_->size();
        pool_size_type stop_  = start_ + static_cast<pool_size_type>(count);
        data_->resize(stop_);
        return PoolItemT(data_, start_, stop_);
    }

  private:
    std::shared_ptr<std::vector<T>> data_{};
};

template<class T>
class Pool<T, Ownership::value, MemSpace::device>
{
  public:
    Pool() = default;

    //! Construct from a host pool, copying data
    explicit Pool(const Pool<T, Ownership::value, MemSpace::host>& host_pool)
    {
        Span<const T> host_data = host_pool.get();
        if (!host_data.empty())
        {
            data_.resize(host_data.size());
            data_.copy_to_device(host_data);
        }
    }

    //! Access all elements from this pool
    Span<const T> get() const { return make_span<*data_>; }

  private:
    DeviceVector<T> data_;
};
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "Pool.i.hh"
