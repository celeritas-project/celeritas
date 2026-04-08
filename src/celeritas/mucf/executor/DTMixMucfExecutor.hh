//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/DTMixMucfExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mucf/data/DTMixMucfData.hh"
#include "celeritas/mucf/interactor/DDMucfInteractor.hh"
#include "celeritas/mucf/interactor/DTMucfInteractor.hh"
#include "celeritas/mucf/interactor/TTMucfInteractor.hh"

#include "detail/DDChannelSelector.hh"
#include "detail/DTChannelSelector.hh"
#include "detail/MuonicAtomSelector.hh"
#include "detail/MuonicAtomSpinSelector.hh"
#include "detail/MuonicMoleculeSelector.hh"
#include "detail/TTChannelSelector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct DTMixMucfExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<DTMixMucfData> data;
};

//---------------------------------------------------------------------------//
/*!
 * Execute muon-catalyzed fusion for muonic dd, dt, or tt molecules.
 */
CELER_FUNCTION Interaction
DTMixMucfExecutor::operator()(celeritas::CoreTrackView const& track)
{
    auto phys_step_view = track.physics_step();
    auto elcomp_id = phys_step_view.element();
    CELER_ASSERT(elcomp_id);

    auto const& mat_record = track.material().material_record();
    auto element = mat_record.element_record(elcomp_id);
    CELER_ASSERT(element.atomic_number() == AtomicNumber{1});  // Must be H

    //! \todo Make sure that at this point we selected d or t already

    // Find muCF material ID from PhysMatId
    // Make this a View if ever used beyond this executor
    auto find = [&](PhysMatId matid) -> MucfMatId {
        CELER_EXPECT(matid);
        for (auto i : range(data.mucfmatid_to_matid.size()))
        {
            if (auto const comp_id = MucfMatId{i};
                data.mucfmatid_to_matid[comp_id] == matid)
            {
                return comp_id;
            }
        }
        // MuCF material ID not found
        return MucfMatId{};
    };
    auto const mucf_matid = find(track.material().material_id());
    CELER_ASSERT(mucf_matid);

    auto rng = track.rng();

    // Form d or t muonic atom
    auto muonic_atom = detail::MuonicAtomSelector(
        data.isotopic_fractions[mucf_matid][MucfIsotope::deuterium])(rng);
    auto atom_spin = detail::MuonicAtomSpinSelector(muonic_atom)(rng);

    // {
    // Competing at-rest processes which add to the total track time
    //! \todo Muonic atom transfer
    //! \todo Muonic atom spin flip
    // }

    // Form dd, dt, or tt muonic molecule
    auto [muonic_molecule, cycle_time] = detail::MuonicMoleculeSelector(
        muonic_atom, atom_spin, data.cycle_rates[mucf_matid])(rng);

    // Update track time according to the sampled cycle time
    track.sim().add_time(cycle_time);

    // Fuse molecule and generate secondaries
    auto allocate_secondaries = phys_step_view.make_secondary_allocator();
    Interaction result;
    switch (muonic_molecule)
    {
        case MucfMuonicMolecule::deuterium_deuterium: {
            // Return DD interaction
            DDMucfInteractor interact(
                data,
                detail::DDChannelSelector(mat_record.temperature())(rng),
                allocate_secondaries);
            result = interact(rng);
            break;
        }
        case MucfMuonicMolecule::deuterium_tritium: {
            // Return DT interaction
            DTMucfInteractor interact(
                data, detail::DTChannelSelector()(rng), allocate_secondaries);
            result = interact(rng);
            break;
        }
        case MucfMuonicMolecule::tritium_tritium: {
            // Return TT interaction
            TTMucfInteractor interact(
                data, detail::TTChannelSelector()(rng), allocate_secondaries);
            result = interact(rng);
            break;
        }
        default:
            CELER_ASSERT_UNREACHABLE();
    }

    //! \todo Muon stripping: strip muon from muonic atom secondaries
    // May be added as a separate discrete process in the stepping loop

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
