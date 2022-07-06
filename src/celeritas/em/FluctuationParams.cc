//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/FluctuationParams.cc
//---------------------------------------------------------------------------//
#include "FluctuationParams.hh"

#include <cmath>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with particle and material data.
 */
FluctuationParams::FluctuationParams(SPConstParticles particles,
                                     SPConstMaterials materials)
{
    CELER_EXPECT(materials);
    CELER_EXPECT(particles);

    celeritas::HostVal<FluctuationData> data;

    // Set particle properties
    data.electron_id = particles->find(pdg::electron());
    CELER_VALIDATE(data.electron_id,
                   << "missing electron particle (required for energy loss "
                      "fluctuations)");
    data.electron_mass = particles->get(data.electron_id).mass().value();

    // Loop over materials
    auto urban = make_builder(&data.urban);
    for (auto mat_id : range(MaterialId{materials->size()}))
    {
        const auto mat = materials->get(mat_id);

        // Calculate the parameters for the energy loss fluctuation model (see
        // Geant3 PHYS332 2.4 and Geant4 physics reference manual 7.3.2)
        UrbanFluctuationParameters params;
        const real_type avg_z = mat.electron_density() / mat.number_density();
        params.oscillator_strength[1] = avg_z > 2 ? 2 / avg_z : 0;
        params.oscillator_strength[0] = 1 - params.oscillator_strength[1];
        params.binding_energy[1]      = 1e-5 * ipow<2>(avg_z);
        params.binding_energy[0]
            = std::pow(mat.mean_excitation_energy().value()
                           / std::pow(params.binding_energy[1],
                                      params.oscillator_strength[1]),
                       1 / params.oscillator_strength[0]);
        params.log_binding_energy[1] = std::log(params.binding_energy[1]);
        params.log_binding_energy[0] = std::log(params.binding_energy[0]);
        urban.push_back(params);
    }

    data_ = CollectionMirror<FluctuationData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
