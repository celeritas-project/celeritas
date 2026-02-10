//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/params/AtomicRelaxationParams.cc
//---------------------------------------------------------------------------//
#include "AtomicRelaxationParams.hh"

#include <set>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/detail/Utils.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialParams.hh"  // IWYU pragma: keep
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a vector of element identifiers.
 */
AtomicRelaxationParams::AtomicRelaxationParams(Input const& inp)
    : is_auger_enabled_(inp.is_auger_enabled)
{
    CELER_EXPECT(inp.cutoffs);
    CELER_EXPECT(inp.materials);
    CELER_EXPECT(inp.particles);

    ScopedMem record_mem("AtomicRelaxationParams.construct");

    HostData host_data;

    // Get particle IDs
    host_data.ids.electron = inp.particles->find(pdg::electron());
    host_data.ids.gamma = inp.particles->find(pdg::gamma());
    CELER_VALIDATE(host_data.ids.electron && host_data.ids.gamma,
                   << "missing electron and/or gamma particles "
                      "(required for atomic relaxation)");

    // Find the minimum electron and photon cutoff energy for each element over
    // all materials. This is used to calculate the maximum number of
    // secondaries that could be created in atomic relaxation for each element.
    size_type num_elements = inp.materials->num_elements();
    std::vector<MevEnergy> electron_cutoff(num_elements, max_quantity());
    std::vector<MevEnergy> gamma_cutoff(num_elements, max_quantity());
    for (auto mat_id : range(PhysMatId{inp.materials->num_materials()}))
    {
        // Electron and photon energy cutoffs for this material
        auto cutoffs = inp.cutoffs->get(mat_id);
        auto material = inp.materials->get(mat_id);
        for (auto comp_id : range(ElementComponentId{material.num_elements()}))
        {
            auto el_idx = material.element_id(comp_id).get();
            electron_cutoff[el_idx]
                = min(electron_cutoff[el_idx],
                      cutoffs.energy(host_data.ids.electron));
            gamma_cutoff[el_idx] = min(gamma_cutoff[el_idx],
                                       cutoffs.energy(host_data.ids.gamma));
        }
    }

    // Build elements
    CELER_LOG(status) << "Reading and building atomic relaxation data";
    ScopedTimeLog scoped_time;
    make_builder(&host_data.elements).reserve(num_elements);
    for (auto el_idx : range(num_elements))
    {
        AtomicNumber z = inp.materials->get(ElementId{el_idx}).atomic_number();
        this->append_element(inp.load_data(z),
                             &host_data,
                             electron_cutoff[el_idx],
                             gamma_cutoff[el_idx]);
    }

    // Move to mirrored data, copying to device
    data_ = ParamsDataStore<AtomicRelaxParamsData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
// IMPLEMENTATION
//---------------------------------------------------------------------------//
/*!
 * Convert an element input to a AtomicRelaxElement and store.
 */
void AtomicRelaxationParams::append_element(ImportAtomicRelaxation const& inp,
                                            HostData* data,
                                            MevEnergy electron_cutoff,
                                            MevEnergy gamma_cutoff)
{
    CELER_EXPECT(data);
    AtomicRelaxElement el;

    // Collect all the subshell designators for this element
    std::set<int> designators;
    for (auto const& shell : inp.shells)
    {
        designators.insert(shell.designator);

        // Check that for a given subshell vacancy EADL transition
        // probabilities are normalized so that the sum over all radiative and
        // non-radiative transitions is 1
        real_type norm = 0;
        for (auto const& transition : shell.fluor)
        {
            norm += transition.probability;
            designators.insert(transition.initial_shell);
        }
        for (auto const& transition : shell.auger)
        {
            norm += transition.probability;
            designators.insert(transition.initial_shell);
            designators.insert(transition.auger_shell);
        }
        CELER_ASSERT(soft_equal(real_type(1), norm));
    }

    // Create a mapping of subshell designator to index in the shells array (it
    // is ok for an index to be greater than or equal to the size of the shells
    // array; this just means there is no transition data for that shell)
    std::unordered_map<int, SubshellId> des_to_id;
    size_type index = 0;
    for (auto des : designators)
    {
        des_to_id[des] = SubshellId{index++};
    }
    CELER_ASSERT(des_to_id.size() >= inp.shells.size());

    // Add subshell data
    std::vector<AtomicRelaxSubshell> shells(inp.shells.size());
    for (auto i : range(inp.shells.size()))
    {
        // Get all the transitions for this subshell
        std::vector<ImportAtomicTransition> import_transitions(
            inp.shells[i].fluor.begin(), inp.shells[i].fluor.end());
        if (is_auger_enabled_)
        {
            // Append the non-radiative transitions if Auger effect is enabled
            import_transitions.insert(import_transitions.end(),
                                      inp.shells[i].auger.begin(),
                                      inp.shells[i].auger.end());
        }

        // Add transition data
        std::vector<AtomicRelaxTransition> transitions(
            import_transitions.size());
        for (auto j : range(import_transitions.size()))
        {
            // Find the index in the shells array given the shell designator.
            // If the designator is not found, map it to an invalid value.
            transitions[j].initial_shell
                = des_to_id[import_transitions[j].initial_shell];
            transitions[j].auger_shell
                = des_to_id[import_transitions[j].auger_shell];
            transitions[j].probability = import_transitions[j].probability;
            transitions[j].energy
                = units::MevEnergy(import_transitions[j].energy);
        }
        shells[i].transitions
            = make_builder(&data->transitions)
                  .insert_back(transitions.begin(), transitions.end());
    }
    el.shells
        = make_builder(&data->shells).insert_back(shells.begin(), shells.end());

    // Calculate the maximum possible number of secondaries that could be
    // created in atomic relaxation.
    el.max_secondary = detail::calc_max_secondaries(
        make_const_ref(*data), el.shells, electron_cutoff, gamma_cutoff);

    // Maximum size of the stack used to store unprocessed vacancy subshell IDs
    data->max_stack_size
        = max(data->max_stack_size,
              detail::calc_max_stack_size(make_const_ref(*data), el.shells));

    // Add the elemental data (no data for Z < 6)
    CELER_ASSERT(el || inp.shells.empty());
    make_builder(&data->elements).push_back(el);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
