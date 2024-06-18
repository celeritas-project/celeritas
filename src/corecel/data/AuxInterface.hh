//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/UserInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string_view>

#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//

//! Index for user-added data
using UserId = OpaqueId<struct User_>;

//---------------------------------------------------------------------------//
// INTERFACES
//---------------------------------------------------------------------------//
/*!
 * User-specified state data owned by a single stream.
 *
 * This interface class is strictly to allow polymorphism and dynamic casting.
 */
class UserStateInterface
{
  public:
    //@{
    //! \name Type aliases
    using SPState = std::shared_ptr<UserStateInterface>;
    //@}

  public:
    // Virtual destructor for polymorphism
    virtual ~UserStateInterface();

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    UserStateInterface() = default;
    CELER_DEFAULT_COPY_MOVE(UserStateInterface);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Base class for extensible shared data that has associated state.
 *
 * "User" data can be added to a \c UserParamsInterface at runtime to be passed
 * among multiple classes, and then \c dynamic_cast to the expected type. It
 * needs to supply a factory function for creating the a state instance for
 * multithreaded data on a particular stream and a given memory space.
 */
class UserParamsInterface
{
  public:
    //@{
    //! \name Type aliases
    using UPState = std::unique_ptr<UserStateInterface>;
    //@}

  public:
    // Virtual destructor for polymorphism
    virtual ~UserParamsInterface();

    //! Index of this class instance in its registry
    virtual UserId user_id() const = 0;

    //! Label for the user data
    virtual std::string_view label() const = 0;

    //! Factory function for building multithread state for a stream
    virtual UPState create_state(MemSpace, StreamId, size_type size) const = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    UserParamsInterface() = default;
    CELER_DEFAULT_COPY_MOVE(UserParamsInterface);
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
