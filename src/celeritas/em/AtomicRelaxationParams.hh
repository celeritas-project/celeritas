//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/AtomicRelaxationParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/io/ImportAtomicRelaxation.hh"
#include "celeritas/phys/AtomicNumber.hh"

#include "data/AtomicRelaxationData.hh"

namespace celeritas
{
class CutoffParams;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Data management for the EADL transition data for atomic relaxation.
 */
class AtomicRelaxationParams final
    : public ParamsDataInterface<AtomicRelaxParamsData>
{
  public:
    //@{
    //! Type aliases
    using MevEnergy = units::MevEnergy;
    using ReadData = std::function<ImportAtomicRelaxation(AtomicNumber)>;
    using SPConstCutoffs = std::shared_ptr<CutoffParams const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    //@}

    struct Input
    {
        SPConstCutoffs cutoffs;
        SPConstMaterials materials;
        SPConstParticles particles;
        ReadData load_data;
        bool is_auger_enabled{false};  //!< Whether to produce Auger electrons
    };

  public:
    // Construct with a vector of element identifiers
    explicit AtomicRelaxationParams(Input const& inp);

    //! Access EADL data on the host
    HostRef const& host_ref() const final { return data_.host(); }

    //! Access EADL data on the device
    DeviceRef const& device_ref() const final { return data_.device(); }

  private:
    // Whether to simulate non-radiative transitions
    bool is_auger_enabled_;

    // Host/device storage and reference
    CollectionMirror<AtomicRelaxParamsData> data_;

    // HELPER FUNCTIONS
    using HostData = HostVal<AtomicRelaxParamsData>;
    void append_element(ImportAtomicRelaxation const& inp,
                        HostData* data,
                        MevEnergy electron_cutoff,
                        MevEnergy gamma_cutoff);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
