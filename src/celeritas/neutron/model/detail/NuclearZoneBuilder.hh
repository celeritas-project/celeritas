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

#include "corecel/cont/Span.hh"
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
    // Calculate the nuclear radius
    inline real_type calc_nuclear_radius(AtomicMassNumber a) const;

    // Calculate components of nuclear zone properties
    inline ComponentVec calc_zone_components(IsotopeView const& target) const;

  private:
    //// DATA ////

    // Cascade model configurations and nuclear structure parameters
    CascadeOptions const& options_;

    real_type skin_depth_;
    MevMass neutron_mass_;
    MevMass proton_mass_;

    CollectionBuilder<ZoneComponent> components_;
    CollectionBuilder<NuclearZones, MemSpace::host, IsotopeId> zones_;
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
    auto comp = this->calc_zone_components(target);

    NuclearZones nucl_zones;
    nucl_zones.zones = components_.insert_back(comp.begin(), comp.end());
    zones_.push_back(nucl_zones);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate components of nuclear zone data: the nuclear zone radius, volume,
 * density, Fermi momentum and potential function as in G4NucleiModel and as
 * documented in section 24.2.3 of the Geant4 Physics Reference (release 11.2).
 *
 * The Woods-Saxon potential, \f$ V(r) \f$,
 *
 * \f[
     V(r) = frac{V_{o}}{1 + e^{\frac{r - R}{a}}}
   \f]
 * numerically over the volume from \f$ r_{min} \f$ to \f$ r_{rmax} \f$, where
 * \f$ V_{o}, R, a\f$ are the potential well depth, nuclear radius, and
 * surface thickness (skin depth), respectively.
 */
auto NuclearZoneBuilder::calc_zone_components(IsotopeView const& target) const
    -> ComponentVec
{
    using A = AtomicNumber;
    A const a = target.atomic_mass_number();

    // Calculate nuclear radius
    real_type nuclear_radius = this->calc_nuclear_radius(a);
    real_type skin_ratio = nuclear_radius / skin_depth_;

    // Temporary data for the zone-by-zone density function
    std::vector<real_type> radii;
    std::vector<real_type> integral;

    // Fill the nuclear radius by each zone
    real_type ymin = (a < A{12}) ? 0 : -skin_ratio;

    if (a < A{5})
    {
        // Light ions treated as simple balls
        radii.push_back(nuclear_radius);
        integral.push_back(1);
    }
    else if (a < A{12})
    {
        // Small nuclei have a three-zone Gaussian potential
        real_type gauss_radius = std::sqrt(
            ipow<2>(nuclear_radius) * (1 - 1 / static_cast<real_type>(a.get()))
            + real_type{6.4});

        Integrator integrate_gauss{
            [](real_type r) { return ipow<2>(r) * std::exp(-ipow<2>(r)); }};

        // Precompute y = sqrt(-log(alpha)) where alpha[3] = {0.7, 0.3, 0.01}
        constexpr Real3 y = {0.597223, 1.09726, 2.14597};
        for (auto i : range(y.size()))
        {
            radii.push_back(gauss_radius * y[i]);
            integral.push_back(ipow<3>(gauss_radius)
                               * integrate_gauss(ymin, y[i]));
            ymin = y[i];
        }
    }
    else
    {
        // Heavier nuclei use Woods-Saxon potential: three zones for
        // intermediate nuclei, six zones for heavy (A >= 100)
        Span<real_type const> alpha;
        if (a < A{100})
        {
            static real_type const alpha_i[] = {0.7, 0.3, 0.01};
            alpha = make_span(alpha_i);
        }
        else
        {
            static real_type const alpha_h[] = {0.9, 0.6, 0.4, 0.2, 0.1, 0.05};
            alpha = make_span(alpha_h);
        }

        real_type skin_decay = std::exp(-skin_ratio);
        Integrator integrate_ws{[ws_shift = 2 * skin_ratio](real_type r) {
            return r * (r + ws_shift) / (1 + std::exp(r));
        }};

        for (auto i : range(alpha.size()))
        {
            real_type y = std::log((1 + skin_decay) / alpha[i] - 1);
            radii.push_back(nuclear_radius + skin_depth_ * y);

            integral.push_back(ipow<3>(skin_depth_)
                               * (integrate_ws(ymin, y)
                                  + ipow<2>(skin_ratio)
                                        * std::log((1 + std::exp(-ymin))
                                                   / (1 + std::exp(-y)))));
            ymin = y;
        }
    }

    // Fill the differential nuclear volume by each zone
    constexpr real_type four_thirds_pi = 4 * constants::pi / real_type{3};

    ComponentVec components(radii.size());
    real_type prev_volume{0};
    for (auto i : range(components.size()))
    {
        components[i].radius = radii[i];
        real_type volume = four_thirds_pi * ipow<3>(radii[i]);

        // Save differential volume
        components[i].volume = volume - prev_volume;
        prev_volume = volume;
    }

    // Fill the nuclear zone density, fermi momentum, and potential
    int num_protons = target.atomic_number().get();
    int num_nucleons[] = {num_protons, a.get() - num_protons};
    real_type mass[] = {proton_mass_.value(), neutron_mass_.value()};
    real_type dm[] = {target.proton_loss_energy().value(),
                      target.neutron_loss_energy().value()};
    static_assert(std::size(mass) == ZoneComponent::NucleonArray::size());
    static_assert(std::size(dm) == std::size(mass));

    real_type const total_integral
        = std::accumulate(integral.begin(), integral.end(), real_type{0});

    for (auto i : range(components.size()))
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
                = ipow<2>(components[i].fermi_mom[ptype]) / (2 * mass[ptype])
                  + dm[ptype];
        }
    }
    return components;
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
real_type NuclearZoneBuilder::calc_nuclear_radius(AtomicMassNumber a) const
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
}  // namespace detail
}  // namespace celeritas
