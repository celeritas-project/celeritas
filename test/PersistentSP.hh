//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PersistentSP.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <gtest/gtest.h>

#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Hold a shared pointer across test cases and clean at teardown.
 *
 * Keep a \c static instance of this class inside a test harness (or anywhere);
 * it will register a cleanup function in GoogleTest that will fire before the
 * end of the program.
 *
 * This is needed to manage persistent objects that use static storage
 * duration, such as VecGeom. For those cases, a static object that cleans up
 * VecGeom on teardown may be called \em after VecGeom's static destructors,
 * since static initialization/destruction order is undefined across
 * translation units.
 */
template<class T>
class PersistentSP
{
  public:
    //!@{
    //! \name Type aliases
    using SP = std::shared_ptr<T>;
    //!@}

  public:
    // Constructor registers the environment
    explicit inline PersistentSP(std::string&& desc);
    ~PersistentSP() = default;

    //! Whether a value is stored
    explicit operator bool() const { return static_cast<bool>(env_->ptr); }

    // Replace the pointer
    inline void set(std::string key, SP ptr);

    //! Clear stored value
    inline void clear() { env_->TearDown(); }

    //! Access the key (empty if unset)
    std::string const& key() const { return env_->key; }

    //! Access the pointer (null if unset)
    SP const& value() const { return env_->ptr; }

  private:
    // Only this class should delete, and it shouldn't be moved/copied
    CELER_DELETE_COPY_MOVE(PersistentSP);

    // GTest environment owns the SP throughout the test run
    struct Env final : public ::testing::Environment
    {
        void SetUp() final {}
        void TearDown() final;

        std::string desc;
        std::string key;
        SP ptr;
    };

    Env* env_{nullptr};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Register the environment on construction.
 */
template<class T>
PersistentSP<T>::PersistentSP(std::string&& desc)
{
    // GoogleTest takes ownership of environment; we retain a non-owning
    // pointer
    auto env = std::make_unique<Env>();
    env_ = env.get();
    CELER_LOG(debug) << "Registering persistent " << desc << " cleanup";
    env_->desc = std::move(desc);
    ::testing::AddGlobalTestEnvironment(env.release());
}

//---------------------------------------------------------------------------//
/*!
 * Replace the pointer.
 */
template<class T>
void PersistentSP<T>::set(std::string key, SP ptr)
{
    CELER_EXPECT(!key.empty());
    CELER_EXPECT(ptr);
    CELER_LOG(debug) << "Updating persistent " << env_->desc << " to '" << key
                     << "'";
    env_->key = std::move(key);
    env_->ptr = std::move(ptr);
}

//---------------------------------------------------------------------------//
/*!
 * Reset the shared pointer.
 */
template<class T>
void PersistentSP<T>::Env::TearDown()
{
    auto use_count = this->ptr.use_count();
    if (use_count == 0)
    {
        CELER_LOG(debug) << "Nothing stored in persistent " << desc;
    }
    else if (use_count == 1)
    {
        CELER_LOG(info) << "Clearing persistent " << this->desc << " '"
                        << this->key << "'";
    }
    else
    {
        CELER_LOG(warning) << "Resetting but not destroying persistent "
                           << this->desc << " '" << this->key
                           << "': use_count=" << use_count;
    }
    this->ptr.reset();
    this->key.clear();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
