//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/params/AtomicRelaxationParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>

#include "corecel/Types.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/AtomicRelaxationData.hh"
#include "celeritas/io/ImportAtomicRelaxation.hh"
#include "celeritas/phys/AtomicNumber.hh"

namespace celeritas
{
class CutoffParams;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Data management for the EADL transition data for atomic relaxation.
 *
 * \note The EADL only provides transition probabilities for 6 <= Z <= 100, so
 * there will be no atomic relaxation data for Z < 6. Transitions are only
 * provided for K, L, M, N, and some O shells.
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
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access EADL data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Whether to simulate non-radiative transitions
    bool is_auger_enabled_;

    // Host/device storage and reference
    ParamsDataStore<AtomicRelaxParamsData> data_;

    // HELPER FUNCTIONS
    using HostData = HostVal<AtomicRelaxParamsData>;
    void append_element(ImportAtomicRelaxation const& inp,
                        HostData* data,
                        MevEnergy electron_cutoff,
                        MevEnergy gamma_cutoff);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
