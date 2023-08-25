//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventIOTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "Test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Helpers for testing reading+writing of event files.
 *
 * This defines several particles: proton, d_quark, anti_u_quark, w_minus,
 * gamma; and provides utilities for reading all primaries from a file.
 *
 */
class EventIOTestBase : public Test
{
  public:
    //!@{
    //! \name Type aliases
    using VecPrimary = std::vector<Primary>;
    using ReadFunc = std::function<VecPrimary()>;
    using WriteFunc = std::function<void(VecPrimary const&)>;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    //!@}

  protected:
    struct ReadAllResult
    {
        std::vector<int> pdg;
        std::vector<double> energy;
        std::vector<double> pos;
        std::vector<double> dir;
        std::vector<double> time;
        std::vector<int> event;
        std::vector<int> track;

        void print_expected() const;
    };

    // Create particles
    void SetUp() override;

    // Read all primaries from a file
    ReadAllResult read_all(ReadFunc read_event) const;

    // Write test primaries to a file
    void write_test_event(WriteFunc write_event) const;
    // Read and check the test primaries to a file
    void read_check_test_event(ReadFunc read_event) const;

    // Access particles
    SPConstParticles const& particles() const { return particles_; }

  private:
    SPConstParticles particles_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
