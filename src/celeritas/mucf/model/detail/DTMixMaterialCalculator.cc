//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/model/detail/DTMixMaterialCalculator.cc
//---------------------------------------------------------------------------//
#include "DTMixMaterialCalculator.hh"

#include "corecel/Assert.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with material data.
 *
 * Calculates and caches material-dependent properties needed by the
 * \c DTMixMucfModel . If the material does not contain deuterium and/or
 * tritium the object's operator bool will return false.
 */
DTMixMaterialCalculator::DTMixMaterialCalculator(MaterialView const& material)
    : material_(material)
{
    for (auto elcompid : range(material_.num_elements()))
    {
        auto const& element_view
            = material_.element_record(ElementComponentId{elcompid});
        if (element_view.atomic_number() != AtomicNumber{1})
        {
            // Skip non-hydrogen elements
            continue;
        }

        has_isotope_ = {false, false};
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

            if (auto const atom = this->from_mass_number(mass);
                atom < MucfMuonicAtom::size_)
            {
                // D and/or t isotopes found; calculate properties
                has_isotope_[atom] = true;
                lhd_densities_ = calc_lhd_densities();
                eq_densities_ = calc_equilibrium_densities();
                cycle_times_ = calc_cycle_times(element_view);
            }
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Convert dt mixture densities to units of liquid hydrogen density.
 *
 * Used during cycle time calculations.
 */
DTMixMaterialCalculator::LhdArray DTMixMaterialCalculator::calc_lhd_densities()
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
DTMixMaterialCalculator::EquilibriumArray
DTMixMaterialCalculator::calc_equilibrium_densities()
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
DTMixMaterialCalculator::CycleTimesArray
DTMixMaterialCalculator::calc_cycle_times(ElementView const& element)
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
                    = this->calc_dd_cycle();
                if (has_isotope_[MucfMuonicAtom::tritium])
                {
                    // Calculate cycle times for dt molecules
                    result[MucfMuonicMolecule::deuterium_tritium]
                        = this->calc_dt_cycle();
                }
                break;
            }
            // Calculate cycle times for tt molecules
            case MucfMuonicAtom::tritium: {
                result[MucfMuonicMolecule::tritium_tritium]
                    = this->calc_tt_cycle();
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
Array<real_type, 2> DTMixMaterialCalculator::calc_dd_cycle()
{
    Array<real_type, 2> result;

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
Array<real_type, 2> DTMixMaterialCalculator::calc_dt_cycle()
{
    Array<real_type, 2> result;

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
Array<real_type, 2> DTMixMaterialCalculator::calc_tt_cycle()
{
    Array<real_type, 2> result;

    //! \todo Implement

    // Only F = 1/2 is reactive
    CELER_ENSURE(result[0] >= 0 && result[1] == 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Return \c MucfMuonicAtom from a given atomic mass number.
 */
MucfMuonicAtom DTMixMaterialCalculator::from_mass_number(AtomicMassNumber mass)
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
}  // namespace detail
}  // namespace celeritas
