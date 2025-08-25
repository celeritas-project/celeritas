//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/SurfaceModel.cc
//---------------------------------------------------------------------------//
#include "SurfaceModel.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * Construct with label and model ID.
 *
 * Note that the label, being a string view, must (for now) point to constant
 * memory.
 */
SurfaceModel::SurfaceModel(SurfaceModelId id, std::string_view label)
    : id_{id}, label_{label}
{
    CELER_EXPECT(id_);
    CELER_EXPECT(!label_.empty());
}

//---------------------------------------------------------------------------//
//! Anchored default destructor
SurfaceModel::~SurfaceModel() = default;

//---------------------------------------------------------------------------//
}  // namespace celeritas
