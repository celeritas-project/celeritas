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
MucfMaterialInserter::MucfMaterialInserter(HostVal<DTMixMucfData>* host_data,
                                           inp::MucfScalars const& scalars_)
    : mucfmatid_to_matid_(&host_data->mucfmatid_to_matid)
    , cycle_times_(&host_data->cycle_times)
    , scalars_(scalars_)
{
    CELER_EXPECT(scalars_);
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
    this->clear();
    auto const mat_num_density = material.number_density();

    for (auto elcompid : range(material.num_elements()))
    {
        auto const& element_view
            = material.element_record(ElementComponentId{elcompid});
        if (element_view.atomic_number() != AtomicNumber{1})
        {
            // Skip non-hydrogen elements
            continue;
        }

        // Found hydrogen; calculate quantities for its isotopes
        auto const elem_rel_abundance = material.elements()[elcompid].fraction;
        for (auto el_comp : range(element_view.num_isotopes()))
        {
            auto iso_view
                = element_view.isotope_record(IsotopeComponentId{el_comp});
            auto const atom
                = this->from_mass_number(iso_view.atomic_mass_number());

            CELER_ASSERT(atom < MucfIsotope::size_);
            has_isotope_[atom] = true;
            lhd_densities_[atom] = elem_rel_abundance * mat_num_density
                                   / scalars_.liquid_hydrogen_density.value();
        }

        if (!has_isotope_[MucfIsotope::deuterium]
            && !has_isotope_[MucfIsotope::tritium])
        {
            // No deuterium or tritium found; skip material
            return false;
        }

        // Found hydrogen with deuterium and/or tritium; Calculate quantities
        equilibrium_densities_ = EquilibrateDensitiesCalculator(
            lhd_densities_, material.temperature())();

        // Calculate and insert muCF material data into model data
        mucfmatid_to_matid_.push_back(material.material_id());
        cycle_times_.push_back(this->calc_cycle_times(element_view));
        //! \todo Store mean atom spin flip and transfer times
    }
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Return \c MucfIsotope from a given atomic mass number.
 */
MucfIsotope MucfMaterialInserter::from_mass_number(AtomicMassNumber mass)
{
    if (mass == AtomicMassNumber{1})
    {
        return MucfIsotope::protium;
    }
    if (mass == AtomicMassNumber{2})
    {
        return MucfIsotope::deuterium;
    }
    if (mass == AtomicMassNumber{3})
    {
        return MucfIsotope::tritium;
    }
    return MucfIsotope::size_;
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
MucfMaterialInserter::calc_cycle_times(ElementView const& element)
{
    CELER_EXPECT(element.atomic_number() == AtomicNumber{1});
    CELER_EXPECT(has_isotope_[MucfIsotope::deuterium]
                 || has_isotope_[MucfIsotope::tritium]);
    CELER_EXPECT(lhd_densities_[MucfIsotope::deuterium] > 0
                 || lhd_densities_[MucfIsotope::tritium] > 0);

    CycleTimesArray result;
    for (auto el_comp : range(element.num_isotopes()))
    {
        auto iso_view = element.isotope_record(IsotopeComponentId{el_comp});

        // Select possible muonic atom based on the isotope/element mass number
        auto atom = this->from_mass_number(iso_view.atomic_mass_number());
        switch (atom)
        {
            // Calculate cycle times for dd molecules
            case MucfIsotope::deuterium: {
                result[MucfMuonicMolecule::deuterium_deuterium]
                    = this->calc_dd_cycle(element);
                if (has_isotope_[MucfIsotope::tritium])
                {
                    // Calculate cycle times for dt molecules
                    result[MucfMuonicMolecule::deuterium_tritium]
                        = this->calc_dt_cycle(element);
                }
                break;
            }
            // Calculate cycle times for tt molecules
            case MucfIsotope::tritium: {
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
/*!
 * Clear temporary data before next insertion.
 */
void MucfMaterialInserter::clear()
{
    for (auto& lhd : lhd_densities_)
    {
        lhd = 0;
    }

    for (auto& has_iso : has_isotope_)
    {
        has_iso = false;
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
