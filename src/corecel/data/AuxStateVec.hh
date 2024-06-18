//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/UserStateVec.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "corecel/Macros.hh"

#include "UserInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class UserParamsRegistry;

//---------------------------------------------------------------------------//
/*!
 * Manage single-stream user state data.
 *
 * This class is constructed from a \c UserParamsRegistry after the params are
 * completely added and while the state is being constructed (with its size,
 * etc.). The UserId for an element of this class corresponds to the
 * UserParamsRegistry.
 *
 * This class can be empty either by default or if the given user registry
 * doesn't have any entries.
 */
class UserStateVec
{
  public:
    //@{
    //! \name Type aliases
    using UPState = std::unique_ptr<UserStateInterface>;
    using UPConstState = std::unique_ptr<UserStateInterface const>;
    //@}

  public:
    //! Create without any user data
    UserStateVec() = default;

    // Create from params on a device/host stream
    UserStateVec(UserParamsRegistry const&, MemSpace, StreamId, size_type);

    // Allow moving; copying is prohibited due to unique pointers
    CELER_DEFAULT_MOVE_DELETE_COPY(UserStateVec);

    // Access user state interfaces
    inline UserStateInterface& at(UserId);
    inline UserStateInterface const& at(UserId) const;

    //! Get the number of defined states
    UserId::size_type size() const { return states_.size(); }

  private:
    std::vector<UPState> states_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a mutable item from a state vector efficiently and safely.
 */
template<class S>
S& get(UserStateVec& vec, UserId uid)
{
    static_assert(std::is_base_of_v<UserStateInterface, S>);
    CELER_EXPECT(uid < vec.size());
    auto* ptr = &vec.at(uid);
    CELER_ENSURE(dynamic_cast<S*>(ptr));
    return *static_cast<S*>(ptr);
}

//---------------------------------------------------------------------------//
/*!
 * Get a const item from a state vector efficiently and safely.
 */
template<class S>
S const& get(UserStateVec const& vec, UserId uid)
{
    static_assert(std::is_base_of_v<UserStateInterface, S>);
    CELER_EXPECT(uid < vec.size());
    auto* ptr = &vec.at(uid);
    CELER_ENSURE(dynamic_cast<S const*>(ptr));
    return *static_cast<S const*>(ptr);
}

//---------------------------------------------------------------------------//
// INLINE DEFININTIONS
//---------------------------------------------------------------------------//
/*!
 * Access a mutable user state interface for a given ID.
 */
UserStateInterface& UserStateVec::at(UserId id)
{
    CELER_EXPECT(id < states_.size());
    return *states_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Access a mutable user state interface for a given ID.
 */
UserStateInterface const& UserStateVec::at(UserId id) const
{
    CELER_EXPECT(id < states_.size());
    return *states_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
