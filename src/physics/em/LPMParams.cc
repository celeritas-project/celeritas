//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LPMParams.cc
//---------------------------------------------------------------------------//
#include "LPMParams.hh"

#include <cmath>

#include "base/Algorithms.hh"
#include "physics/base/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared particle and material data.
 */
LPMParams::LPMParams(SPConstParticles particles)
{
    CELER_EXPECT(particles);

    HostValue host_data;

    // Save particle properties
    host_data.electron_mass = value_as<units::MevMass>(
        particles->get(particles->find(pdg::electron())).mass());

    // Build a table of LPM functions, G(s) and \phi(s) in the range
    // s = [0, data->limit_s_lpm()] with the 1/data->inv_delta_lpm() interval
    auto      lpm_table  = make_builder(&host_data.lpm_table);
    size_type num_points = host_data.inv_delta() * host_data.s_limit() + 1;
    lpm_table.reserve(num_points);

    for (auto s_point : range(num_points))
    {
        real_type s_hat  = s_point / host_data.inv_delta();
        auto      s_data = this->compute_lpm_data(s_hat);
        lpm_table.push_back(s_data);
    }

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<LPMData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Compute LPM table data.
 *
 * Precompute the LPM suppression functions \f$ G(s) \f$ and \f$ \phi(s) \f$ by
 * using a piecewise approximation with simple analytic functions.
 *
 * See section 10.2.2 of the Geant4 Physics Reference Manual and
 * ComputeLPMGsPhis in G4eBremsstrahlungRelModel and G4PairProductionRelModel.
 */
auto LPMParams::compute_lpm_data(real_type x) -> MigdalData
{
    MigdalData data;

    if (x < 0.01)
    {
        data.phi = 6 * x * (1 - constants::pi * x);
        data.g   = 12 * x - 2 * data.phi;
    }
    else
    {
        real_type x2 = ipow<2>(x);
        real_type x3 = x * x2;
        real_type x4 = ipow<2>(x2);

        // use Stanev approximation: for \psi(s) and compute G(s)
        if (x < 0.415827)
        {
            data.phi = 1
                       - std::exp(-6 * x * (1 + x * (3 - constants::pi))
                                  + x3 / (0.623 + 0.796 * x + 0.658 * x2));
            real_type psi = 1
                            - std::exp(-4 * x
                                       - 8 * x2
                                             / (1 + 3.936 * x + 4.97 * x2
                                                - 0.05 * x3 + 7.5 * x4));
            data.g = 3 * psi - 2 * data.phi;
        }
        else if (x < 1.55)
        {
            data.phi = 1
                       - std::exp(-6 * x * (1 + x * (3 - constants::pi))
                                  + x3 / (0.623 + 0.796 * x + 0.658 * x2));
            data.g = std::tanh(-0.160723 + 3.755030 * x - 1.798138 * x2
                               + 0.672827 * x3 - 0.120772 * x4);
        }
        else
        {
            data.phi = 1 - 0.011905 / x4;
            if (x < 1.9156)
            {
                data.g = std::tanh(-0.160723 + 3.755030 * x - 1.798138 * x2
                                   + 0.672827 * x3 - 0.120772 * x4);
            }
            else
            {
                data.g = 1 - 0.023065 / x4;
            }
        }
    }

    return data;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
