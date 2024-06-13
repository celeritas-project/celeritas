//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/detail/NuclearZoneBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <numeric>

#include "corecel/math/Integrator.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/neutron/data/NeutronInelasticData.hh"
#include "celeritas/phys/AtomicNumber.hh"

#include "../CascadeOptions.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct NuclearZoneData for NeutronInelasticModel.
 */
class NuclearZoneBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using AtomicMassNumber = AtomicNumber;
    using MevMass = units::MevMass;
    using Energy = units::MevEnergy;
    using ComponentVec = std::vector<ZoneComponent>;
    using Data = HostVal<NuclearZoneData>;
    //!@}

  public:
    // Construct with cascade options and shared data
    inline NuclearZoneBuilder(CascadeOptions const& options,
                              NeutronInelasticScalars const& scalars,
                              Data* data);

    // Construct nuclear zone data for a target (isotope)
    inline void operator()(IsotopeView const& target);

  private:
    // Calculate the number of nuclear zones
    inline size_type num_nuclear_zones(AtomicMassNumber a);

    // Calculate the nuclear radius
    inline real_type calc_nuclear_radius(AtomicMassNumber a);

    // Calculate components of nuclear zone properties
    inline void
    calc_zone_component(IsotopeView const& target, ComponentVec& components);

    // Integrate the Woods-Saxon potential in [rmin, rmax]
    inline real_type integrate_woods_saxon(real_type rmin,
                                           real_type rmax,
                                           real_type radius) const;

    // Integrate the Gaussoan potential in [rmin, rmax]
    inline real_type
    integrate_gaussian(real_type rmin, real_type rmax, real_type radius) const;

  private:
    //// DATA ////

    // Cascade model configurations and nuclear structure parameters
    CascadeOptions const& options_;

    real_type skin_depth_;
    MevMass neutron_mass_;
    MevMass proton_mass_;

    CollectionBuilder<ZoneComponent> components_;
    CollectionBuilder<NuclearZones, MemSpace::host, IsotopeId> zones_;

    //// COMMON PROPERTIES ////

    static CELER_CONSTEXPR_FUNCTION real_type half() { return 0.5; }
    static CELER_CONSTEXPR_FUNCTION real_type four_thirds_pi()
    {
        return 4 * constants::pi / real_type{3};
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with cascade options and data.
 */
NuclearZoneBuilder::NuclearZoneBuilder(CascadeOptions const& options,
                                       NeutronInelasticScalars const& scalars,
                                       Data* data)
    : options_(options)
    , skin_depth_(0.611207 * options.radius_scale)
    , neutron_mass_(scalars.neutron_mass)
    , proton_mass_(scalars.proton_mass)
    , components_(&data->components)
    , zones_(&data->zones)
{
    CELER_EXPECT(options_);
    CELER_EXPECT(data);
}

//---------------------------------------------------------------------------//
/*!
 * Construct nuclear zone data for a single isotope.
 */
void NuclearZoneBuilder::operator()(IsotopeView const& target)
{
    AtomicNumber a = target.atomic_mass_number();
    size_type num_zones = this->num_nuclear_zones(a);

    std::vector<ZoneComponent> comp(num_zones);
    this->calc_zone_component(target, comp);

    NuclearZones nucl_zones;
    nucl_zones.zones = components_.insert_back(comp.begin(), comp.end());
    zones_.push_back(nucl_zones);
}

//---------------------------------------------------------------------------//
/*!
 * Return the number of nuclear zones.
 */
size_type NuclearZoneBuilder::num_nuclear_zones(AtomicMassNumber a)
{
    if (a < AtomicMassNumber{5})
    {
        return 1;
    }
    if (a < AtomicMassNumber{100})
    {
        return 3;
    }
    return 6;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate components of nuclear zone data: the nuclear zone radius, volume,
 * density, Fermi momentum and potential function as in G4NucleiModel and as
 * documented in section 24.2.3 of the Geant4 Physics Reference (release 11.2).
 */
void NuclearZoneBuilder::calc_zone_component(IsotopeView const& target,
                                             ComponentVec& components)
{
    AtomicNumber a = target.atomic_mass_number();

    // Calculate nuclear radius
    real_type nuclear_radius = this->calc_nuclear_radius(a);

    real_type skin_ratio = nuclear_radius / skin_depth_;
    real_type skin_decay = std::exp(-skin_ratio);

    // Temporary data for the zone-by-zone density function
    size_type num_of_zones = components.size();
    std::vector<real_type> integral(num_of_zones);

    // Fill the nuclear radius by each zone
    auto amass = a.get();
    real_type ymin = (amass < 12) ? 0 : -skin_ratio;

    if (amass < 5)
    {
        // Light ions treated as simple balls
        components[0].radius = nuclear_radius;
        integral[0] = 1;
    }
    else if (amass < 12)
    {
        // Small nuclei have a three-zone Gaussian potential
        real_type gauss_radius = std::sqrt(
            ipow<2>(nuclear_radius) * (1 - 1 / static_cast<real_type>(amass))
            + real_type{6.4});
        // Precompute y = sqrt(-log(alpha)) where alpha[3] = {0.7, 0.3, 0.01}
        constexpr Real3 y = {0.597223, 1.09726, 2.14597};
        for (auto i : range(num_of_zones))
        {
            components[i].radius = gauss_radius * y[i];
            integral[i] = this->integrate_gaussian(ymin, y[i], gauss_radius);
            ymin = y[i];
        }
    }
    else if (amass < 100)
    {
        // Intermediate nuclei have a three-zone Woods-Saxon potential
        constexpr Real3 alpha = {0.7, 0.3, 0.01};
        for (auto i : range(num_of_zones))
        {
            real_type y = std::log((1 + skin_decay) / alpha[i] - 1);
            components[i].radius = nuclear_radius + skin_depth_ * y;
            integral[i] = this->integrate_woods_saxon(ymin, y, nuclear_radius);
            ymin = y;
        }
    }
    else
    {
        // Heavy nuclei have a six-zone Woods-Saxon potential
        constexpr Array<real_type, 6> alpha = {0.9, 0.6, 0.4, 0.2, 0.1, 0.05};
        for (auto i : range(num_of_zones))
        {
            real_type y = std::log((1 + skin_decay) / alpha[i] - 1);
            components[i].radius = nuclear_radius + skin_depth_ * y;
            integral[i] = this->integrate_woods_saxon(ymin, y, nuclear_radius);
            ymin = y;
        }
    }
    real_type total_integral
        = std::accumulate(integral.begin(), integral.end(), real_type{0});

    // Fill the nuclear volume by each zone
    for (auto i : range(num_of_zones))
    {
        components[i].volume = ipow<3>(components[i].radius);
        if (i > 0)
        {
            components[i].volume -= ipow<3>(components[i - 1].radius);
        }
        components[i].volume *= this->four_thirds_pi();
    }

    // Fill the nuclear zone density, fermi momentum, and potential
    int num_protons = target.atomic_number().get();
    int num_nucleons[] = {num_protons, a.get() - num_protons};
    real_type mass[] = {proton_mass_.value(), neutron_mass_.value()};
    real_type dm[] = {target.proton_loss_energy().value(),
                      target.neutron_loss_energy().value()};
    static_assert(std::size(mass) == ZoneComponent::NucleonArray::size());
    static_assert(std::size(dm) == std::size(mass));

    for (auto i : range(num_of_zones))
    {
        for (auto ptype : range(ZoneComponent::NucleonArray::size()))
        {
            components[i].density[ptype]
                = static_cast<real_type>(num_nucleons[ptype]) * integral[i]
                  / (total_integral * components[i].volume);
            components[i].fermi_mom[ptype]
                = options_.fermi_scale
                  * std::cbrt(components[i].density[ptype]);
            components[i].potential[ptype]
                = this->half() * ipow<2>(components[i].fermi_mom[ptype])
                      / mass[ptype]
                  + dm[ptype];
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the nuclear radius (R) computed from the atomic mass number (A).
 *
 * For \f$ A > 4 \f$, the nuclear radius with two parameters takes the form,
 * \f[
     R = [ 1.16 * A^{1/3} - 1.3456 / A^{1/3} ] \cdot R_{scale}
   \f]
 * where \f$ R_{scale} \f$ is a configurable parameter in [femtometer], while
 * \f$ R = 1.2 A^{1/3} \cdot R_{scale} \f$ (default) with a single parameter.
 */
real_type NuclearZoneBuilder::calc_nuclear_radius(AtomicMassNumber a)
{
    // Nuclear radius computed from A
    real_type nuclear_radius{0};
    auto amass = a.get();

    if (amass > 4)
    {
        real_type cbrt_a = std::cbrt(static_cast<real_type>(amass));
        real_type par_a = (options_.use_two_params ? 1.16 : 1.2);
        real_type par_b = (options_.use_two_params ? -1.3456 : 0);
        nuclear_radius = options_.radius_scale
                         * (par_a * cbrt_a + par_b / cbrt_a);
    }
    else
    {
        nuclear_radius = options_.radius_small
                         * (amass == 4 ? options_.radius_alpha : 1);
    }

    return nuclear_radius;
}

//---------------------------------------------------------------------------//
/*!
 * Integrate the Woods-Saxon potential, \f$ V(r) \f$,
 *
 * \f[
     V(r) = frac{V_{o}}{1 + e^{\frac{r - R}{a}}}
   \f]
 * numerically over the volume from \f$ r_{min} \f$ to \f$ r_{rmax} \f$, where
 * \f$ V_{o}, R, a\f$ are the potential well depth, nuclear radius, and
 * surface thickness (skin depth), respectively.
 */
real_type
NuclearZoneBuilder::integrate_woods_saxon(real_type rmin,
                                          real_type rmax,
                                          real_type nuclear_radius) const
{
    real_type skin_ratio = nuclear_radius / skin_depth_;
    Integrator integrate_ws{[ws_shift = 2 * skin_ratio](real_type r) {
        return r * (r + ws_shift) / (1 + std::exp(r));
    }};

    real_type result = integrate_ws(rmin, rmax);

    return ipow<3>(skin_depth_)
           * (result
              + ipow<2>(skin_ratio)
                    * std::log((1 + std::exp(-rmin)) / (1 + std::exp(-rmax))));
}

//---------------------------------------------------------------------------//
/*!
 * Integrate the Gaussian potential function from \f$ r_1 \f$ to \f$ r_2 \f$.
 */
real_type NuclearZoneBuilder::integrate_gaussian(real_type rmin,
                                                 real_type rmax,
                                                 real_type gauss_radius) const
{
    Integrator integrate_gauss{
        [](real_type r) { return ipow<2>(r) * std::exp(-ipow<2>(r)); }};

    return ipow<3>(gauss_radius) * integrate_gauss(rmin, rmax);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
