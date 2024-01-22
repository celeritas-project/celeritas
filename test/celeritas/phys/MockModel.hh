//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/MockModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "celeritas/Types.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Mock model.
 *
 * The model is applicable to a single particle type and energy range. Its
 * "interact" simply calls a test-code-provided callback with the model ID.
 */
class MockModel final : public Model
{
  public:
    //!@{
    //! \name Type aliases
    using BarnMicroXs = Quantity<units::Barn>;
    using ModelCallback = std::function<void(ActionId)>;
    using VecMicroXs = std::vector<BarnMicroXs>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    //!@}

    struct Input
    {
        ActionId id;
        SPConstMaterials materials;
        Applicability applic;
        ModelCallback cb;
        VecMicroXs xs;
    };

  public:
    explicit MockModel(Input data);
    SetApplicability applicability() const final;
    MicroXsBuilders micro_xs(Applicability range) const final;
    void execute(CoreParams const&, CoreStateHost&) const final;
    void execute(CoreParams const&, CoreStateDevice&) const final;
    ActionId action_id() const final { return data_.id; }
    std::string label() const final;
    std::string description() const final;

  private:
    Input data_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
