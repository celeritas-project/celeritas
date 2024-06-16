//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/UserParamsRegistry.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "UserInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage user-added parameter classes.
 *
 * An instance of this class can be added to shared problem data so that users
 * can share arbitrary information between parts of the code and create
 * independent state data for each stream.
 */
class UserParamsRegistry
{
  public:
    //!@{
    //! \name Type aliases
    using SPParams = std::shared_ptr<UserParamsInterface>;
    using SPConstParams = std::shared_ptr<UserParamsInterface const>;
    //!@}

  public:
    // Default constructor
    UserParamsRegistry() = default;

    //// CONSTRUCTION ////

    //! Get the next available ID
    UserId next_id() const { return UserId(params_.size()); }

    // Register user parameters
    void insert(SPParams params);

    //! Get the number of defined params
    UserId::size_type size() const { return params_.size(); }

    // Access params at the given ID
    inline SPParams const& at(UserId);
    inline SPConstParams at(UserId) const;

    // Get the label corresponding to user params
    inline std::string const& id_to_label(UserId id) const;

    // Find the ID corresponding to an label
    UserId find(std::string const& label) const;

  private:
    std::vector<SPParams> params_;
    std::vector<std::string> labels_;
    std::unordered_map<std::string, UserId> user_ids_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Access mutable params at the given ID.
 */
auto UserParamsRegistry::at(UserId id) -> SPParams const&
{
    CELER_EXPECT(id < params_.size());
    return params_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Access params at the given ID.
 */
auto UserParamsRegistry::at(UserId id) const -> SPConstParams
{
    CELER_EXPECT(id < params_.size());
    return params_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the label corresponding to user params.
 */
std::string const& UserParamsRegistry::id_to_label(UserId id) const
{
    CELER_EXPECT(id < params_.size());
    return labels_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
