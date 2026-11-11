//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file minimal/minimal.cc
//---------------------------------------------------------------------------//
#include <memory>
#include <celeritas/Constants.hh>
#include <celeritas/phys/PDGNumber.hh>
#include <celeritas/phys/ParticleParams.hh>
#include <corecel/io/Logger.hh>
#include <corecel/math/Quantity.hh>
#include <corecel/sys/Device.hh>
#include <corecel/sys/ScopedMpiInit.hh>

using celeritas::ParticleParams;
using celeritas::PDGNumber;
using celeritas::value_as;
using celeritas::units::ElementaryCharge;
using celeritas::units::MevMass;

namespace
{
//---------------------------------------------------------------------------//
std::shared_ptr<ParticleParams> make_particles()
{
    constexpr auto zero = celeritas::zero_quantity();
    constexpr auto stable = celeritas::constants::stable_decay_constant;

    ParticleParams::Input defs;
    defs.push_back({"electron",
                    celeritas::pdg::electron(),
                    MevMass{0.5109989461},
                    ElementaryCharge{-1},
                    stable});
    defs.push_back({"gamma", celeritas::pdg::gamma(), zero, zero, stable});
    defs.push_back({"neutron",
                    PDGNumber{2112},
                    MevMass{939.565413},
                    zero,
                    1.0 / (879.4 * celeritas::units::second)});

    return std::make_shared<ParticleParams>(std::move(defs));
}

//---------------------------------------------------------------------------//
}  // namespace

int main(int argc, char* argv[])
{
    // Initialize MPI
    celeritas::ScopedMpiInit scoped_mpi(&argc, &argv);

    // Initialize GPU
    celeritas::activate_device();

    // Create particle definitions (copies to GPU if available)
    auto particles = make_particles();

    // Find the identifier for a neutron and make sure it exists
    celeritas::ParticleId pid = particles->find(PDGNumber{2112});
    CELER_ASSERT(pid);

    // Get a particle "view" with properties about the neutron
    auto neutron = particles->get(pid);
    CELER_LOG(info) << "Neutron has a mass of "
                    << value_as<MevMass>(neutron.mass()) << "MeV / c^2"
                    << " = " << native_value_from(neutron.mass()) << " g";
    CELER_LOG(info) << "Its decay constant is " << neutron.decay_constant()
                    << " /s";
    CELER_LOG(info) << "This example doesn't do anything useful! Sorry!";

    return 0;
}
