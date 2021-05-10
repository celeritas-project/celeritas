//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MockModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Model.hh"
#include <functional>

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
    using Applicability         = celeritas::Applicability;
    using ModelId               = celeritas::ModelId;
    using ModelCallback         = std::function<void(ModelId)>;
    //!@}

  public:
    MockModel(ModelId id, Applicability applic, ModelCallback cb);
    SetApplicability applicability() const final;
    void             interact(const DeviceInteractRefs&) const final;
    ModelId          model_id() const final { return id_; }
    std::string      label() const final;

  private:
    ModelId       id_;
    Applicability applic_;
    ModelCallback cb_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
