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
                                           inp::MucfPhysics const& data)
    : mucfmatid_to_matid_(&host_data->mucfmatid_to_matid)
    , isotopic_fractions_(&host_data->isotopic_fractions)
    , cycle_times_(&host_data->cycle_times)
    , data_(data)
{
    CELER_EXPECT(data_);

    // Initialize interpolators for cycle time tables
    for (auto const& cycle_data : data_.cycle_rates)
    {
        // Use emplace to avoid copy/move of InterpolatorHelper objects
        interpolators_.emplace(
            std::pair{cycle_data.type, cycle_data.spin_state}, cycle_data.rate);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Insert material information if applicable.
 */
bool MucfMaterialInserter::operator()(MaterialView const& material)
{
    using LhdArray = EquilibrateDensitiesSolver::LhdArray;

    MaterialFractionsArray isotopic_fractions;
    CycleTimesArray cycle_times;
    LhdArray lhd_densities{};

    auto from_mass_number = [&](AtomicMassNumber mass) -> MucfIsotope {
        auto it = mass_isotope_map_.find(mass);
        CELER_ENSURE(it != mass_isotope_map_.end());
        return it->second;
    };

    for (auto elcompid : range(material.num_elements()))
    {
        auto const& element_view
            = material.element_record(ElementComponentId{elcompid});
        if (element_view.atomic_number() != AtomicNumber{1})
        {
            // Skip non-hydrogen elements
            continue;
        }

        // Found hydrogen; Check isotopes
        auto const elem_rel_abundance = material.elements()[elcompid].fraction;
        for (auto el_comp : range(element_view.num_isotopes()))
        {
            auto iso_view
                = element_view.isotope_record(IsotopeComponentId{el_comp});
            auto const atom = from_mass_number(iso_view.atomic_mass_number());
            auto const iso_frac = element_view.isotopes()[el_comp].fraction;

            // Cache density for this hydrogen isotope
            isotopic_fractions[atom] = iso_frac;
            lhd_densities[atom]
                = iso_frac
                  * (elem_rel_abundance * material.number_density()
                     / data_.scalars.liquid_hydrogen_density.value());
        }
    }

    if (!lhd_densities[MucfIsotope::deuterium]
        && !lhd_densities[MucfIsotope::tritium])
    {
        // No deuterium or tritium densities; skip material
        return false;
    }

    // Found d and/or t, calculate and insert data into collection

    auto equilibrium_densities
        = EquilibrateDensitiesSolver(lhd_densities)(material.temperature());

    if (lhd_densities[MucfIsotope::deuterium])
    {
        cycle_times[MucfMuonicMolecule::deuterium_deuterium]
            = this->calc_dd_cycle(equilibrium_densities,
                                  material.temperature());
    }
    if (lhd_densities[MucfIsotope::tritium])
    {
        cycle_times[MucfMuonicMolecule::tritium_tritium] = this->calc_tt_cycle(
            equilibrium_densities, material.temperature());
    }
    if (lhd_densities[MucfIsotope::deuterium]
        && lhd_densities[MucfIsotope::tritium])
    {
        cycle_times[MucfMuonicMolecule::deuterium_tritium]
            = this->calc_dt_cycle(equilibrium_densities,
                                  material.temperature());
    }

    // Add muCF material to the model's host/device data
    mucfmatid_to_matid_.push_back(material.material_id());
    isotopic_fractions_.push_back(std::move(isotopic_fractions));
    cycle_times_.push_back(std::move(cycle_times));

    //! \todo Store mean atom spin flip and transfer times

    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate dd muonic molecules cycle times.
 *
 * F = 1/2 and F = 3/2 are the reactive spin states for dd fusion.
 */
MucfMaterialInserter::MoleculeCycles
MucfMaterialInserter::calc_dd_cycle(EquilibriumArray const& eq_dens,
                                    real_type const temperature)
{
    using IsoProt = EquilibrateDensitiesSolver::MucfIsoprotologueMolecule;
    using CTT = inp::CycleTableType;
    using units::HalfSpinInt;

    auto const& dd_dens = eq_dens[IsoProt::deuterium_deuterium];

    auto const& dd_1_over_2_interpolate
        = this->interpolator(CTT::deuterium_deuterium, HalfSpinInt{1});
    auto const& dd_3_over_2_interpolate
        = this->interpolator(CTT::deuterium_deuterium, HalfSpinInt{3});

    MoleculeCycles result;
    result[0] = real_type{1}
                / (dd_dens * dd_1_over_2_interpolate(temperature));  // F = 1/2
    result[1] = real_type{1}
                / (dd_dens * dd_3_over_2_interpolate(temperature));  // F = 3/2

    CELER_ENSURE(result[0] >= 0 && result[1] >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate dt muonic molecules cycle times.
 *
 * F = 0 and F = 1 are the reactive spin states for dt fusion.
 */
MucfMaterialInserter::MoleculeCycles
MucfMaterialInserter::calc_dt_cycle(EquilibriumArray const& eq_dens,
                                    real_type const temperature)
{
    CELER_EXPECT(temperature > 0);

    using IsoProt = EquilibrateDensitiesSolver::MucfIsoprotologueMolecule;
    using CTT = inp::CycleTableType;
    using units::HalfSpinInt;

    auto const& dd_dens = eq_dens[IsoProt::deuterium_deuterium];
    auto const& dt_dens = eq_dens[IsoProt::deuterium_tritium];
    auto const& hd_dens = eq_dens[IsoProt::protium_deuterium];

    // F = 0 interpolators
    auto const& hd0_interpolate
        = this->interpolator(CTT::protium_deuterium, HalfSpinInt{0});
    auto const& dd0_interpolate
        = this->interpolator(CTT::deuterium_deuterium, HalfSpinInt{0});
    auto const& dt0_interpolate
        = this->interpolator(CTT::deuterium_tritium, HalfSpinInt{0});
    // F = 1 interpolators
    auto const& hd1_interpolate
        = this->interpolator(CTT::protium_deuterium, HalfSpinInt{2});
    auto const& dd1_interpolate
        = this->interpolator(CTT::deuterium_deuterium, HalfSpinInt{2});
    auto const& dt1_interpolate
        = this->interpolator(CTT::deuterium_tritium, HalfSpinInt{2});

    // Interpolate over rates, store final cycle time (1/rate)
    MoleculeCycles result;
    result[0] = real_type{1}
                / (hd_dens * hd0_interpolate(temperature)
                   + dd_dens * dd0_interpolate(temperature)
                   + dt_dens * dt0_interpolate(temperature));  // F = 0
    result[1] = real_type{1}
                / (hd_dens * hd1_interpolate(temperature)
                   + dd_dens * dd1_interpolate(temperature)
                   + dt_dens * dt1_interpolate(temperature));  // F = 1

    CELER_ENSURE(result[0] >= 0 && result[1] >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate tt muonic molecules cycle times.
 *
 * F = 1/2 is the only reactive spin state for tt fusion.
 */
MucfMaterialInserter::MoleculeCycles
MucfMaterialInserter::calc_tt_cycle(EquilibriumArray const& eq_dens,
                                    real_type const temperature)
{
    using IsoProt = EquilibrateDensitiesSolver::MucfIsoprotologueMolecule;
    using CTT = inp::CycleTableType;
    using units::HalfSpinInt;

    auto const& tt_dens = eq_dens[IsoProt::tritium_tritium];
    auto const& tt_interpolate
        = this->interpolator(CTT::tritium_tritium, HalfSpinInt{1});

    MoleculeCycles result;
    result[0] = real_type{1} / (tt_dens * tt_interpolate(temperature));

    CELER_ENSURE(result[0] >= 0 && result[1] == 0);
    return result;
}

//---------------------------------------------------------------------------//
InterpolatorHelper const&
MucfMaterialInserter::interpolator(inp::CycleTableType type,
                                   units::HalfSpinInt spin) const
{
    auto it = interpolators_.find({type, spin});
    CELER_ASSERT(it != interpolators_.end());
    return it->second;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
