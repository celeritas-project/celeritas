//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalProcess.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Unique optical process associated with a given optical model.
 */
class OpticalProcess
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstModel = std::shared_ptr<OpticalModel const>;
    using SPConstImported = std::shared_ptr<ImportedOpticalProcesses const>;
    using StepLimitBuilder = std::unique_ptr<GenericGridBuilder const>;
    using ActionIdIter = RangeIter<ActionId>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    //!@}

  public:

    inline CELER_FUNCTION
    OpticalProcess(ImportProcessClass ipc,
                   SPConstMaterials materials,
                   SPConstImported shared_data);

    //! Get the interaction cross sections for optical photons
    std::vector<OpticalValueGridId>
    step_limits(GenericGridInserter& inserter) const;

    virtual SPConstModel build_model(ActionIdIter start_id) const = 0;
    virtual std::string label() const = 0;

  protected:
    ImportedOpticalProcessAdapter imported_;
    SPConstMaterials materials_;
};

template <class OpticalModelImpl>
class OpticalProcessInstance : public OpticalProcess
{
  public:
    inline CELER_FUNCTION
    OpticalProcessInstance(SPConstMaterials materials,
                           SPConstImported shared_data)
        : OpticalProcess(OpticalModelImpl::process_class(), materials, shared_data)
    {}

    //! Create unique model associated with this process
    SPConstModel build_model(ActionIdIter start_id) const override final
    {
        return std::make_shared<OpticalModelImpl>(*start_id++, *materials_);
    }

    //! Name of the optical process
    std::string label() const override final
    {
        return OpticalModelImpl::process_label();
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

OpticalProcess::OpticalProcess(ImportProcessClass ipc,
                               SPConstMaterials materials,
                               SPConstImported shared_data)
    : imported_(shared_data, ipc), materials_(std::move(materials))
{
    CELER_EXPECT(materials_);
}


std::vector<OpticalValueGridId>
OpticalProcess::step_limits(GenericGridInserter& inserter) const
{
    return imported_.step_limits(inserter, *materials_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
