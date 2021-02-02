//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MockModel.cc
//---------------------------------------------------------------------------//
#include "MockModel.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
MockModel::MockModel(ModelId id, Applicability applic, ModelCallback cb)
    : id_(id), applic_(std::move(applic)), cb_(std::move(cb))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(applic_);
    CELER_EXPECT(cb_);
}

auto MockModel::applicability() const -> SetApplicability
{
    return {applic_};
}

void MockModel::interact(const ModelInteractPointers&) const
{
    // Inform calling test code that we've been launched
    cb_(this->model_id());
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
