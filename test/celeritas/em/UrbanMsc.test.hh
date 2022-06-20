//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/UrbanMsc.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/mat/MaterialData.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/PhysicsData.hh"
#include "celeritas/random/RngData.hh"

#include "detail/Macros.hh"

using namespace celeritas;

namespace celeritas_test
{
using GeometryParamsRef
    = GeoParamsData<Ownership::const_reference, MemSpace::device>;
using GeometryStateRef = GeoStateData<Ownership::reference, MemSpace::device>;

using ParticleParamsRef
    = ParticleParamsData<Ownership::const_reference, MemSpace::device>;
using ParticleStateRef
    = ParticleStateData<Ownership::reference, MemSpace::device>;

using MaterialParamsRef
    = MaterialParamsData<Ownership::const_reference, MemSpace::device>;

using PhysicsParamsRef
    = PhysicsParamsData<Ownership::const_reference, MemSpace::device>;
using PhysicsStateRef
    = PhysicsStateData<Ownership::reference, MemSpace::device>;

using UrbanMscDataRef
    = UrbanMscData<Ownership::const_reference, MemSpace::device>;

using RngDeviceRef = RngStateData<Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Test parameters
struct MscTestParams
{
    size_type nstates{1};         //! number of states to test
    Real3     position{0, 0, 0};  //! particle starting position
    Real3     direction{1, 0, 0}; //! particle starting direction
};

//! Input data
struct PhysTestInit
{
    MaterialId material;
    ParticleId particle;
};

struct MscTestInput
{
    std::vector<ParticleTrackInitializer> init_part;
    std::vector<PhysTestInit>             init_phys;

    GeometryParamsRef geometry_params;
    MaterialParamsRef material_params;
    ParticleParamsRef particle_params;
    PhysicsParamsRef  physics_params;

    GeometryStateRef geometry_states;
    ParticleStateRef particle_states;
    PhysicsStateRef  physics_states;

    RngDeviceRef    rng_states;
    UrbanMscDataRef msc_data;
    MscTestParams   test_param;
};

//! Output results
struct MscTestOutput
{
    real_type true_path{};
    real_type geom_path{};
    real_type lateral_arm{};
    real_type psi_value{};
    real_type mom_xdir{};
    real_type phi_correl{};
};

//! Expected results from Geant4 TestEM15 at test energies (MeV)
static const double energy[] = {100, 10, 1, 1e-1, 1e-2, 1e-3};

static const double g4_geom_path[]
    = {7.9736, 1.3991e-1, 2.8978e-3, 9.8068e-5, 1.9926e-6, 1.7734e-7};

static const double g4_true_path[]
    = {8.8845, 1.5101e-1, 3.082e-3, 1.0651e-4, 2.1776e-6, 2.5102e-7};

static const double g4_lateral_arm[]
    = {0, 4.1431e-2, 7.6514e-4, 3.0315e-5, 6.4119e-7, 1.2969e-7};

static const double g4_psi_value[]
    = {0, 0.28637, 0.25691, 0.29862, 0.31131, 0.63142};

static const double g4_mom_xdir[]
    = {1, 0.83511, 0.86961, 0.84, 0.83257, 0.46774};

static const double g4_phi_correl[]
    = {0, 0.37091, 0.32647, 0.35153, 0.34172, 0.58865};

//---------------------------------------------------------------------------//
//! Run on device and return results
std::vector<MscTestOutput> msc_test(MscTestInput input);

#if !CELER_USE_DEVICE
inline std::vector<MscTestOutput> msc_test(MscTestInput)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
//! Return results of multiple scattering test variables
inline CELER_FUNCTION MscTestOutput
calc_output(const MscStep& step_result, const MscInteraction& sample_result)
{
    MscTestOutput result;

    // True and geometrical path length
    result.true_path = sample_result.step_length;
    result.geom_path = step_result.geom_path;

    // Lateral displacement
    double disp_y      = sample_result.displacement[1];
    double disp_z      = sample_result.displacement[2];
    double lateral_arm = sqrt(ipow<2>(disp_y) + ipow<2>(disp_z));
    result.lateral_arm = lateral_arm;

    // Psi variable
    result.psi_value = std::atan(lateral_arm / step_result.geom_path);

    // Angle along the line of flight
    result.mom_xdir = sample_result.direction[0];

    // Phi correlation
    if (lateral_arm > 0)
    {
        result.phi_correl = (disp_y * sample_result.direction[1]
                             + disp_z * sample_result.direction[2])
                            / lateral_arm;
    }

    return result;
}

//! Calculate mean of test variables
inline MscTestOutput calc_mean(const std::vector<MscTestOutput> output)
{
    CELER_EXPECT(output.size() > 0);

    MscTestOutput result;

    unsigned int nsamples        = output.size();
    double       sum_true_path   = 0;
    double       sum_geom_path   = 0;
    double       sum_lateral_arm = 0;
    double       sum_psi_value   = 0;
    double       sum_mom_xdir    = 0;
    double       sum_phi_correl  = 0;

    for (auto i : celeritas::range(nsamples))
    {
        sum_geom_path += output[i].geom_path;
        sum_true_path += output[i].true_path;
        sum_lateral_arm += output[i].lateral_arm;
        sum_psi_value += output[i].psi_value;
        sum_mom_xdir += output[i].mom_xdir;
        sum_phi_correl += output[i].phi_correl;
    }

    // Mean of test variables
    result.geom_path   = sum_geom_path / nsamples;
    result.true_path   = sum_true_path / nsamples;
    result.lateral_arm = sum_lateral_arm / nsamples;
    result.psi_value   = sum_psi_value / nsamples;
    result.mom_xdir    = sum_mom_xdir / nsamples;
    result.phi_correl  = sum_phi_correl / nsamples;

    return result;
}

//! Compare results against the Geant4 TestEM15 (extended) example
inline void check_result(const std::vector<MscTestOutput> result)
{
    CELER_EXPECT(result.size() > 0);

    // Tolerance of the relative error with respect to Geant4: percent
    constexpr double tolerance = 0.01;
    for (auto i : celeritas::range(result.size()))
    {
        using namespace celeritas_test;
        EXPECT_SOFT_NEAR(g4_geom_path[i], result[i].geom_path, tolerance);
        EXPECT_SOFT_NEAR(g4_true_path[i], result[i].true_path, tolerance);
        EXPECT_SOFT_NEAR(g4_lateral_arm[i], result[i].lateral_arm, tolerance);
        EXPECT_SOFT_NEAR(g4_psi_value[i], result[i].psi_value, tolerance);
        EXPECT_SOFT_NEAR(g4_mom_xdir[i], result[i].mom_xdir, tolerance);
        EXPECT_SOFT_NEAR(g4_phi_correl[i], result[i].phi_correl, tolerance);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
