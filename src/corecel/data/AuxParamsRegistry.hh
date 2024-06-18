//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AuxParamsRegistry.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "AuxInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage auxiliary-added parameter classes.
 *
 * An instance of this class can be added to shared problem data so that users
 * (and other parts of Celeritas) can share arbitrary information between parts
 * of the code and create independent state data for each stream.
 */
class AuxParamsRegistry
{
  public:
    //!@{
    //! \name Type aliases
    using SPParams = std::shared_ptr<AuxParamsInterface>;
    using SPConstParams = std::shared_ptr<AuxParamsInterface const>;
    //!@}

  public:
    // Default constructor
    AuxParamsRegistry() = default;

    //// CONSTRUCTION ////

    //! Get the next available ID
    AuxId next_id() const { return AuxId(params_.size()); }

    // Register auxiliary parameters
    void insert(SPParams params);

    //! Get the number of defined params
    AuxId::size_type size() const { return params_.size(); }

    // Access params at the given ID
    inline SPParams const& at(AuxId);
    inline SPConstParams at(AuxId) const;

    // Get the label corresponding to auxiliary params
    inline std::string const& id_to_label(AuxId id) const;

    // Find the ID corresponding to an label
    AuxId find(std::string const& label) const;

  private:
    std::vector<SPParams> params_;
    std::vector<std::string> labels_;
    std::unordered_map<std::string, AuxId> aux_ids_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Access mutable params at the given ID.
 */
auto AuxParamsRegistry::at(AuxId id) -> SPParams const&
{
    CELER_EXPECT(id < params_.size());
    return params_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Access params at the given ID.
 */
auto AuxParamsRegistry::at(AuxId id) const -> SPConstParams
{
    CELER_EXPECT(id < params_.size());
    return params_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the label corresponding to auxiliary params.
 */
std::string const& AuxParamsRegistry::id_to_label(AuxId id) const
{
    CELER_EXPECT(id < params_.size());
    return labels_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
