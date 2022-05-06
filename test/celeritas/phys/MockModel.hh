//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/MockModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "celeritas/Types.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Mock model.
 *
 * The model is applicable to a single particle type and energy range. Its
 * "interact" simply calls a test-code-provided callback with the model ID.
 */
class MockModel final : public celeritas::Model
{
  public:
    //!@{
    //! Type aliases
    using Applicability = celeritas::Applicability;
    using ActionId      = celeritas::ActionId;
    using ModelCallback = std::function<void(ActionId)>;
    //!@}

  public:
    MockModel(ActionId id, Applicability applic, ModelCallback cb);
    SetApplicability applicability() const final;
    void             execute(CoreHostRef const&) const final;
    void             execute(CoreDeviceRef const&) const final;
    ActionId         action_id() const final { return id_; }
    std::string      label() const final;
    std::string      description() const final;

  private:
    ActionId      id_;
    Applicability applic_;
    ModelCallback cb_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
