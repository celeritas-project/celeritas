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
    , cycle_times_(&host_data->cycle_times)
    , data_(data)
{
    CELER_EXPECT(data_);

    // Initialize interpolators for cycle time tables
    for (auto const& cycle_data : data_.cycle_rates)
    {
        InterpolatorHelper interp(cycle_data.rate);
        interpolators_.insert(
            {{cycle_data.type, cycle_data.spin_state}, interp});
    }
}

//---------------------------------------------------------------------------//
/*!
 * Insert material information if applicable.
 *
 * Calculates and caches material-dependent properties needed by the
 * \c DTMixMucfModel . If the material does not contain deuterium and/or
 * tritium the operator will return false.
 *
 * * This is designed to work with the user's material definition being either:
 * - Single element, multiple isotopes (H element, with H, d, and t isotopes);
 * or
 * - Multiple elements, single isotope each (separate H, d, and t elements).
 */
bool MucfMaterialInserter::operator()(MaterialView const& material)
{
    using LhdArray = EquilibrateDensitiesCalculator::LhdArray;

    CycleTimesArray cycle_times;
    LhdArray lhd_densities{};

    auto from_mass_number = [&](AtomicMassNumber mass) -> MucfIsotope {
        auto it = mass_isotope_map_.find(mass);
        return (it != mass_isotope_map_.end()) ? it->second
                                               : MucfIsotope::size_;
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
            CELER_ASSERT(atom < MucfIsotope::size_);

            // Cache density for hydrogen isotope
            lhd_densities[atom]
                = elem_rel_abundance * material.number_density()
                  / data_.scalars.liquid_hydrogen_density.value();
        }
    }

    if (!lhd_densities[MucfIsotope::deuterium]
        && !lhd_densities[MucfIsotope::tritium])
    {
        // No deuterium or tritium densities; skip material
        return false;
    }

    // Found d and/or t, calculate and insert data into collection

    auto equilibrium_densities = EquilibrateDensitiesCalculator(
        lhd_densities, material.temperature())();

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
    MoleculeCycles result{0, 0};

    //! \todo Implement

    // Reactive states are F = 1/2 and F = 3/2
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

    using IsoProt = MucfIsoprotologueMolecule;
    using CTT = inp::CycleTableType;
    using units::HalfSpinInt;

    auto const& dd_dens = eq_dens[IsoProt::deuterium_deuterium];
    auto const& dt_dens = eq_dens[IsoProt::deuterium_tritium];
    auto const& hd_dens = eq_dens[IsoProt::protium_deuterium];

    // F = 0 interpolators
    auto dd0_interpolate
        = interpolators_.find({CTT::deuterium_deuterium, HalfSpinInt{0}})->second;
    auto dt0_interpolate
        = interpolators_.find({CTT::deuterium_tritium, HalfSpinInt{0}})->second;
    auto hd0_interpolate
        = interpolators_.find({CTT::protium_deuterium, HalfSpinInt{0}})->second;

    // F = 1 interpolators
    auto dd1_interpolate
        = interpolators_.find({CTT::deuterium_deuterium, HalfSpinInt{2}})->second;
    auto dt1_interpolate
        = interpolators_.find({CTT::deuterium_tritium, HalfSpinInt{2}})->second;
    auto hd1_interpolate
        = interpolators_.find({CTT::protium_deuterium, HalfSpinInt{2}})->second;

    MoleculeCycles result{1, 2};
#if 0
    // F = 0
    result[0] = dd_dens * dd0_interpolate(temperature)
                + dt_dens * dt0_interpolate(temperature)
                + hd_dens * hd0_interpolate(temperature);

    // F = 1
    result[1] = dd_dens * dd1_interpolate(temperature)
                + dt_dens * dt1_interpolate(temperature)
                + hd_dens * hd1_interpolate(temperature);
#endif

    // Reactive states are F = 0 and F = 1
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
    MoleculeCycles result{0, 0};

    //! \todo Implement

    // Only F = 1/2 is reactive
    CELER_ENSURE(result[0] >= 0 && result[1] == 0);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
