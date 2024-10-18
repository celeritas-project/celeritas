//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighModel.cc
//---------------------------------------------------------------------------//
#include "RayleighModel.hh"

#include <iostream>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Constants.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/optical/MfpBuilder.hh"

#include "../MaterialParams.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from imported data.
 */
RayleighModel::RayleighModel(ActionId id,
                             ImportedModelAdapter imported,
                             Input input)
    : Model(id, "optical-rayleigh", "interact by optical Rayleigh")
    , imported_(std::move(imported))
    , properties_(std::move(input.properties))
    , rayleigh_(std::move(input.rayleigh))
{
    CELER_EXPECT(properties_.size() == imported_.num_materials());
    CELER_EXPECT(rayleigh_.size() == properties_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Build the mean free paths for the model.
 */
void RayleighModel::build_mfps(OpticalMaterialId mat, MfpBuilder& build) const
{
    using namespace celeritas::constants;

    if (auto const& mfp = imported_.mfp(mat))
    {
        build(mfp);
    }
    else if (auto const& rayl = rayleigh_[mat.get()])
    {
        ImportPhysicsVector const& r_index
            = properties_[mat.get()].refractive_index;

        constexpr real_type hbarc = hbar_planck * c_light;

        // Einstein-Smoluchowski formula
        real_type scale_factor = rayl.scale_factor;
        real_type beta_t = rayl.compressibility;
        real_type temperature = rayl.temperature;

        real_type c1 = scale_factor * beta_t * temperature * k_boltzmann
                       / (6 * pi);

        ImportPhysicsVector calculated_mfp{
            ImportPhysicsVectorType::free,
            r_index.x,
            std::vector<double>(r_index.y.size(), 0)};
        for (auto i : range(r_index.x.size()))
        {
            real_type energy = r_index.x[i];
            real_type n = r_index.y[i];

            CELER_ASSERT(n > 1);

            real_type c2 = ipow<4>(energy / hbarc);
            real_type c3 = ipow<2>((ipow<2>(n) - 1) * (ipow<2>(n) + 2) / 3);

            CELER_ASSERT(c2 > 0);
            CELER_ASSERT(c3 > 0);

            calculated_mfp.y[i] = 1 / (c1 * c2 * c3);
        }

        build(calculated_mfp);
    }
    else
    {
        CELER_LOG(warning)
            << "Could not construct optical Rayleigh MFP table for "
               "optical material "
            << mat.get() << " since its imported data was invalid";
        build();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the host.
 */
void RayleighModel::step(CoreParams const&, CoreStateHost&) const
{
    CELER_NOT_IMPLEMENTED("optical core physics");
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the device.
 */
#if !CELER_USE_DEVICE
void RayleighModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
