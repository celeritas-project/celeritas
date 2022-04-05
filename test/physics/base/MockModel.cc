//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MockModel.cc
//---------------------------------------------------------------------------//
#include "MockModel.hh"

#include <sstream>

namespace celeritas_test
{
//---------------------------------------------------------------------------//
MockModel::MockModel(ActionId id, Applicability applic, ModelCallback cb)
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

void MockModel::interact(const HostInteractRef&) const
{
    // Shouldn't be called?
}

void MockModel::interact(const DeviceInteractRef&) const
{
    // Inform calling test code that we've been launched
    cb_(this->action_id());
}

std::string MockModel::label() const
{
    std::ostringstream os;
    os << "MockModel(" << id_.get() << ", p=" << applic_.particle.get()
       << ", emin=" << applic_.lower.value()
       << ", emax=" << applic_.upper.value() << ")";
    return os.str();
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
