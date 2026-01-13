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
#include "detail/DTMixMuonicAtomSelector.hh"
#include "detail/DTMixMuonicMoleculeSelector.hh"
#include "detail/MuonicAtomSpinSelector.hh"
#include "detail/MuonicMoleculeSpinSelector.hh"
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

    auto element = track.material().material_record().element_record(elcomp_id);
    CELER_ASSERT(element.atomic_number() == AtomicNumber{1});  // Must be H

    auto rng = track.rng();

    // Muon decay may compete against other "actions" in this executor
    real_type const decay_len{};  //! \todo Set muon decay interaction length

    // Form d or t muonic atom
    detail::DTMixMuonicAtomSelector form_atom;
    auto muonic_atom = form_atom(rng);

    // Select atom spin via a helper class
    detail::MuonicAtomSpinSelector select_atom_spin(muonic_atom);

    // {
    // Competing at-rest processes which add to the total track time
    //! \todo Muonic atom transfer
    //! \todo Muonic atom spin flip
    // }

    // Form dd, dt, or tt muonic molecule
    detail::DTMixMuonicMoleculeSelector form_muonic_molecule;
    auto muonic_molecule = form_muonic_molecule(rng);

    // Select molecule spin
    detail::MuonicMoleculeSpinSelector select_molecule_spin(muonic_molecule);
    auto const molecule_spin = select_molecule_spin(rng);

    // Find muCF material ID from PhysMatId
    // Make this a View if ever used beyond this executor
    auto find = [&](PhysMatId matid) -> MuCfMatId {
        CELER_EXPECT(matid);
        for (auto i : range(data.mucfmatid_to_matid.size()))
        {
            if (auto const comp_id = MuCfMatId{i};
                data.mucfmatid_to_matid[comp_id] == matid)
            {
                return comp_id;
            }
        }
        // MuCF material ID not found
        return MuCfMatId{};
    };

    // Load cycle time for the selected molecule
    auto const mucf_matid = find(track.material().material_id());
    CELER_ASSERT(mucf_matid);
    auto const cycle_time
        = data.cycle_times[mucf_matid][muonic_molecule][molecule_spin];
    CELER_ASSERT(cycle_time > 0);

    // Check if muon decays before fusion happens
    real_type const mucf_len = cycle_time * track.sim().step_length();
    if (decay_len < mucf_len)
    {
        // Muon decays and halts the interaction
        //! \todo Update track time and return muon decay interactor
    }

    //! \todo Correct track time update? Or should be done in Interactors?
    track.sim().add_time(cycle_time);

    // Fuse molecule and generate secondaries
    //! \todo Maybe move the channel selectors into the interactors
    auto allocate_secondaries = phys_step_view.make_secondary_allocator();
    Interaction result;
    switch (muonic_molecule)
    {
        case MucfMuonicMolecule::deuterium_deuterium: {
            // Return DD interaction
            DDMucfInteractor interact(
                data, detail::DDChannelSelector()(rng), allocate_secondaries);
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
