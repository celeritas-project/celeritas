//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/AlongStepTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "../GlobalTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Run one or more tracks with the same starting conditions for a single step.
 *
 * This high-level test *only* executes on the host so we can extract detailed
 * information from the states.
 */
class AlongStepTestBase : virtual public GlobalTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using size_type = size_type;
    using real_type = real_type;
    using Real3     = Real3;

    using MevEnergy = units::MevEnergy;
    //!@}

    struct Input
    {
        ParticleId particle_id;
        MevEnergy  energy{0};
        Real3      position{0, 0, 0};
        Real3      direction{0, 0, 1};
        real_type  time{0};
        real_type  phys_mfp{1}; //!< Number of MFP to collision

        explicit operator bool() const
        {
            return particle_id && energy >= zero_quantity() && time >= 0
                   && phys_mfp > 0;
        }
    };

    struct RunResult
    {
        real_type   eloss{};        //!< Energy loss / MeV
        real_type   displacement{}; //!< Distance from start to end points
        real_type   angle{};        //!< Dot product of in/out direction
        real_type   time{};         //!< Change in time
        real_type   step{};         //!< Physical step length
        std::string action;         //!< Most likely action to take next

        void print_expected() const;
    };

    RunResult run(const Input&, size_type num_tracks = 1);
};

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
