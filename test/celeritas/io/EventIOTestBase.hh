//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventIOTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "celeritas/io/EventIOInterface.hh"
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
 */
class EventIOTestBase : public Test
{
  public:
    //!@{
    //! \name Type aliases
    using VecPrimary = std::vector<Primary>;
    using Reader = EventReaderInterface;
    using Writer = EventWriterInterface;
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

        void print_expected() const;
    };

    // Create particles
    void SetUp() override;

    // Read all primaries from a file
    ReadAllResult read_all(Reader& read_event) const;

    // Write test primaries to a file
    void write_test_event(Writer& write_event) const;
    // Read and check the test primaries to a file
    void read_check_test_event(Reader& read_event) const;

    // Access particles
    SPConstParticles const& particles() const { return particles_; }

  private:
    SPConstParticles particles_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
