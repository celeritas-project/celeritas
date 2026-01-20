//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/model/detail/MucfMaterialInserter.cc
//---------------------------------------------------------------------------//
#include "MucfMaterialInserter.hh"

#include "corecel/Assert.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with \c DTMixMucfModel model data.
 */
MucfMaterialInserter::MucfMaterialInserter(HostVal<DTMixMucfData>* host_data)
    : mucfmatid_to_matid_(&host_data->mucfmatid_to_matid)
    , cycle_times_(&host_data->cycle_times)
{
    CELER_EXPECT(host_data);
}

//---------------------------------------------------------------------------//
/*!
 * Insert material information if applicable.
 *
 * Calculates and caches material-dependent properties needed by the
 * \c DTMixMucfModel . If the material does not contain deuterium and/or
 * tritium the operator will return false.
 */
bool MucfMaterialInserter::operator()(MaterialView const& material)
{
    for (auto elcompid : range(material.num_elements()))
    {
        auto const& element_view
            = material.element_record(ElementComponentId{elcompid});
        if (element_view.atomic_number() != AtomicNumber{1})
        {
            // Skip non-hydrogen elements
            continue;
        }

        // Found hydrogen; Check for d and t isotopes
        IsotopeChecker has_isotope{false, false};
        for (auto el_comp : range(element_view.num_isotopes()))
        {
            auto iso_view
                = element_view.isotope_record(IsotopeComponentId{el_comp});
            auto mass = iso_view.atomic_mass_number();
            if (mass == AtomicMassNumber{1})
            {
                // Skip protium
                continue;
            }

            auto const atom = this->from_mass_number(mass);
            if (atom < MucfMuonicAtom::size_)
            {
                // Mark d or t isotope as found
                has_isotope[atom] = true;
            }
        }

        if (!has_isotope[MucfMuonicAtom::deuterium]
            && !has_isotope[MucfMuonicAtom::tritium])
        {
            // No deuterium or tritium found; skip material
            return false;
        }

        // Temporary data needed to calculate model data, such as cycle times
        lhd_densities_ = this->calc_lhd_densities(element_view);
        equilibrium_densities_ = this->calc_equilibrium_densities(element_view);

        // Calculate and insert muCF material data into model data
        mucfmatid_to_matid_.push_back(material.material_id());
        cycle_times_.push_back(
            this->calc_cycle_times(element_view, has_isotope));
        //! \todo Store mean atom spin flip and transfer times
    }
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Return \c MucfMuonicAtom from a given atomic mass number.
 */
MucfMuonicAtom MucfMaterialInserter::from_mass_number(AtomicMassNumber mass)
{
    if (mass == AtomicMassNumber{2})
    {
        return MucfMuonicAtom::deuterium;
    }
    if (mass == AtomicMassNumber{3})
    {
        return MucfMuonicAtom::tritium;
    }
    return MucfMuonicAtom::size_;
}

//---------------------------------------------------------------------------//
/*!
 * Convert dt mixture densities to units of liquid hydrogen density.
 *
 * Used during cycle time calculations.
 */
MucfMaterialInserter::LhdArray
MucfMaterialInserter::calc_lhd_densities(ElementView const&)
{
    LhdArray result;

    //! \todo Implement

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate dt mixture densities after reaching thermodynamical
 * equilibrium.
 *
 * Used during cycle time calculations.
 */
MucfMaterialInserter::EquilibriumArray
MucfMaterialInserter::calc_equilibrium_densities(ElementView const&)
{
    EquilibriumArray result;

    //! \todo Implement

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate fusion mean cycle times.
 *
 * This is designed to work with the user's material definition being either:
 * - Single element, multiple isotopes (H element, with H, d, and t isotopes);
 * or
 * - Multiple elements, single isotope each (separate H, d, and t elements).
 */
MucfMaterialInserter::CycleTimesArray
MucfMaterialInserter::calc_cycle_times(ElementView const& element,
                                       IsotopeChecker const& has_isotope)
{
    CycleTimesArray result;
    for (auto el_comp : range(element.num_isotopes()))
    {
        auto iso_view = element.isotope_record(IsotopeComponentId{el_comp});

        // Select possible muonic atom based on the isotope/element mass number
        auto atom = this->from_mass_number(iso_view.atomic_mass_number());
        switch (atom)
        {
            // Calculate cycle times for dd molecules
            case MucfMuonicAtom::deuterium: {
                result[MucfMuonicMolecule::deuterium_deuterium]
                    = this->calc_dd_cycle(element);
                if (has_isotope[MucfMuonicAtom::tritium])
                {
                    // Calculate cycle times for dt molecules
                    result[MucfMuonicMolecule::deuterium_tritium]
                        = this->calc_dt_cycle(element);
                }
                break;
            }
            // Calculate cycle times for tt molecules
            case MucfMuonicAtom::tritium: {
                result[MucfMuonicMolecule::tritium_tritium]
                    = this->calc_tt_cycle(element);
                break;
            }
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate dd muonic molecules cycle times from material properties and grid
 * data.
 *
 * Cycle times for dd molecules come from F = 0 and F = 1 spin states.
 */
MucfMaterialInserter::MoleculeCycles
MucfMaterialInserter::calc_dd_cycle(ElementView const&)
{
    MoleculeCycles result;

    //! \todo Implement

    // Reactive states are F = 0 and F = 1
    CELER_ENSURE(result[0] >= 0 && result[1] >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate dt muonic molecules cycle times from material properties and grid
 * data.
 *
 * Cycle times for dt molecules come from F = 1/2 and F = 3/2 spin states.
 */
MucfMaterialInserter::MoleculeCycles
MucfMaterialInserter::calc_dt_cycle(ElementView const&)
{
    MoleculeCycles result;

    //! \todo Implement

    // Reactive states are F = 1/2 and F = 3/2
    CELER_ENSURE(result[0] >= 0 && result[1] >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate tt muonic molecules cycle times from material properties and grid
 * data.
 *
 * Cycle times for tt molecules come only from the F = 1/2 spin state.
 */
MucfMaterialInserter::MoleculeCycles
MucfMaterialInserter::calc_tt_cycle(ElementView const&)
{
    MoleculeCycles result;

    //! \todo Implement

    // Only F = 1/2 is reactive
    CELER_ENSURE(result[0] >= 0 && result[1] == 0);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
