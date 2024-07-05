//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AuxParamsRegistry.cc
//---------------------------------------------------------------------------//
#include "AuxParamsRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Register auxiliary parameters.
 */
void AuxParamsRegistry::insert(SPParams params)
{
    auto label = std::string{params->label()};
    CELER_VALIDATE(!label.empty(), << "auxiliary params label is empty");

    auto id = params->aux_id();
    CELER_VALIDATE(id == this->next_id(),
                   << "incorrect id {" << id.unchecked_get()
                   << "} for auxiliary params '" << label << "' (should be {"
                   << this->next_id().get() << "})");

    auto iter_inserted = aux_ids_.insert({label, id});
    CELER_VALIDATE(iter_inserted.second,
                   << "duplicate auxiliary params label '" << label << "'");

    params_.push_back(std::move(params));
    labels_.push_back(std::move(label));

    CELER_ENSURE(aux_ids_.size() == params_.size());
    CELER_ENSURE(labels_.size() == params_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Find the auxiliary params corresponding to an label.
 */
AuxId AuxParamsRegistry::find(std::string const& label) const
{
    auto iter = aux_ids_.find(label);
    if (iter == aux_ids_.end())
        return {};
    return iter->second;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
