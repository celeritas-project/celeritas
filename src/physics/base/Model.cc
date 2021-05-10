//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Model.cc
//---------------------------------------------------------------------------//
#include "Model.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Default virtual destructor for polymorphic deletion.
Model::~Model() = default;

//---------------------------------------------------------------------------//
/*!
 * Default to "not implemented" host interaction.
 */
void Model::interact(const HostInteractRefs&) const
{
    CELER_NOT_IMPLEMENTED("host interactions");
}

//---------------------------------------------------------------------------//
} // namespace celeritas
