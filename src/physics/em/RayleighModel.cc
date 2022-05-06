//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RayleighModel.cc
//---------------------------------------------------------------------------//
#include "RayleighModel.hh"

#include "base/Assert.hh"
#include "base/CollectionBuilder.hh"
#include "base/Quantity.hh"
#include "base/Range.hh"
#include "base/detail/RangeImpl.hh"
#include "physics/base/Applicability.hh"
#include "physics/base/PDGNumber.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/Units.hh"
#include "physics/material/ElementView.hh"
#include "sim/Types.hh"

#include "detail/RayleighData.hh"
#include "generated/RayleighInteract.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
RayleighModel::RayleighModel(ActionId              id,
                             const ParticleParams& particles,
                             const MaterialParams& materials)
{
    CELER_EXPECT(id);

    HostValue host_ref;

    host_ref.ids.action = id;
    host_ref.ids.gamma  = particles.find(pdg::gamma());
    CELER_VALIDATE(host_ref.ids.gamma,
                   << "missing gamma particles (required for "
                   << this->description() << ")");

    this->build_data(&host_ref, materials);

    // Move to mirrored data, copying to device
    mirror_ = CollectionMirror<detail::RayleighData>{std::move(host_ref)};

    CELER_ENSURE(this->mirror_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto RayleighModel::applicability() const -> SetApplicability
{
    Applicability rayleigh_scattering;
    rayleigh_scattering.particle = this->host_ref().ids.gamma;
    rayleigh_scattering.lower    = zero_quantity();
    rayleigh_scattering.upper    = units::MevEnergy{1e+8};

    return {rayleigh_scattering};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void RayleighModel::execute(CoreDeviceRef const& data) const
{
    generated::rayleigh_interact(this->device_ref(), data);
}

void RayleighModel::execute(CoreHostRef const& data) const
{
    generated::rayleigh_interact(this->host_ref(), data);
}

//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId RayleighModel::action_id() const
{
    return this->host_ref().ids.action;
}

//---------------------------------------------------------------------------//
/*!
 * Construct RayleighParameters for all the elements in the problem.
 */
void RayleighModel::build_data(HostValue* data, const MaterialParams& materials)
{
    // Number of elements
    unsigned int num_elements = materials.num_elements();

    // Build data for available elements
    auto params = make_builder(&data->params);
    params.reserve(num_elements);

    for (auto el_id : range(ElementId{num_elements}))
    {
        params.push_back(RayleighModel::get_el_parameters(
            materials.get(el_id).atomic_number()));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Build elemental angular scattering parameters.
 *
 * Tabulated data of the form factor based on modified fit formulas from
 * Dermott E. Cullen, Nucl. Instrum. Meth. Phys. Res. B v.101, (4),499-510.
 * The angular distribution of coherently scattered photons is a product of
 * Rayleigh scattering (\em R) and a correction factor (\em f),
 * \f[
    \Phi (E, \cos) = R(\cos\theta) f(E, \cos)
     R = [1 + cos^{2}]
     f = [FF(E, cos) + AS(E)]^{2}
   \f]
 * where \em cos is the cosine of the photon scattering angle, \em E is the
 * incident photon energy, \em FF is the form factor (unit-less) and \em AS
 * is the anomalous scattering factor.
 *
 * Excerpted from G4RayleighAngularGenerator.cc of Geant4 6.10.
 * Parameters for Z = 0 are dropped as they are zeros and not used.
 * Reshaped as [el][param][3] with params.T.reshape((100, 3, 3)),
 * then updated 'n' with params[:,2,:] -= 1
 */
auto RayleighModel::get_el_parameters(AtomicNumber z) -> const ElScatParams&
{
    CELER_EXPECT(z > 0);
    static const ElScatParams el_params[]
        = {{{0.00000e+00, 1.00000e+00, 0.00000e+00},
            {1.53728e-16, 1.10561e-15, 6.89413e-17},
            {5.57834e+00, 2.99983e+00, 3.00000e+00}},
           {{2.00000e+00, 2.00000e+00, 0.00000e+00},
            {2.95909e-16, 3.50254e-16, 2.11456e-17},
            {2.91446e+00, 5.63093e+00, 3.00000e+00}},
           {{5.21459e+00, 3.77240e+00, 1.30091e-02},
            {1.95042e-15, 1.56836e-16, 2.47782e-17},
            {6.59547e+00, 2.85593e+00, 4.94686e+00}},
           {{1.02817e+01, 2.17924e+00, 3.53906e+00},
            {6.24521e-16, 7.86286e-15, 7.01557e-17},
            {9.70700e+00, 6.93420e-01, 3.10265e+00}},
           {{3.66207e+00, 1.19967e+01, 9.34125e+00},
            {4.69459e-17, 2.27060e-16, 1.01544e-15},
            {2.97317e+00, 1.37911e+01, 6.87177e+00}},
           {{3.63903e+00, 1.77772e+01, 1.45838e+01},
            {3.13940e-17, 7.28454e-16, 1.76177e-16},
            {3.00593e+00, 6.03995e+00, 1.10509e+01}},
           {{3.71155e+00, 2.35265e+01, 2.17619e+01},
            {2.38808e-17, 4.54123e-16, 1.28191e-16},
            {2.93206e+00, 7.89527e+00, 1.10472e+01}},
           {{3.65165e+01, 2.37970e+01, 3.68644e+00},
            {3.59428e-16, 8.03792e-17, 1.80511e-17},
            {7.10644e+00, 1.21929e+01, 2.90597e+00}},
           {{3.43548e+00, 3.99937e+01, 3.75709e+01},
            {1.29470e-17, 4.91833e-16, 1.96803e-16},
            {2.97743e+00, 3.93354e+00, 4.34338e+00}},
           {{3.40045e+00, 4.67748e+01, 4.98248e+01},
            {1.01182e-17, 1.45891e-16, 3.16753e-16},
            {3.04641e+00, 4.59461e+00, 5.33072e+00}},
           {{2.87811e+00, 6.00000e+01, 5.81219e+01},
            {6.99543e-18, 1.71829e-16, 1.21362e-15},
            {3.30202e+00, 2.98033e+00, 1.76777e+00}},
           {{3.35541e+00, 6.86446e+01, 7.20000e+01},
            {6.51380e-18, 3.90707e-15, 6.63660e-17},
            {3.19399e+00, 7.45780e-01, 6.90099e+00}},
           {{3.21141e+00, 8.17887e+01, 8.39999e+01},
            {5.24063e-18, 2.76487e-15, 8.42625e-17},
            {3.27399e+00, 1.67629e+00, 4.58323e+00}},
           {{2.95234e+00, 9.80000e+01, 9.50477e+01},
            {4.12831e-18, 4.34500e-16, 1.01935e-16},
            {3.41690e+00, 1.31840e+01, 3.26372e+00}},
           {{3.02524e+00, 1.12000e+02, 1.09975e+02},
            {4.22067e-18, 6.80131e-16, 1.34162e-16},
            {3.04829e+00, 7.88775e+00, 2.30050e+00}},
           {{1.26146e+02, 1.28000e+02, 1.85351e+00},
            {2.12802e-16, 4.04186e-16, 1.87076e-18},
            {1.21745e+00, 1.21809e+01, 4.69179e+00}},
           {{1.75044e+02, 9.67939e+01, 1.71623e+01},
            {3.27035e-16, 8.95703e-17, 2.76259e-17},
            {1.03523e+01, 3.51627e+00, 1.36980e+00}},
           {{1.62000e+02, 1.62000e+02, 0.00000e+00},
            {2.27705e-16, 3.32136e-16, 1.22170e-16},
            {8.49760e-01, 1.27677e+01, 2.68167e+00}},
           {{2.96833e+02, 6.15575e+01, 2.60927e+00},
            {1.86943e-15, 1.38470e-17, 1.66059e-18},
            {6.19050e-01, 8.53727e+00, 4.28070e+00}},
           {{3.00994e+02, 9.64218e+01, 2.58422e+00},
            {8.10577e-16, 4.16869e-17, 1.76249e-18},
            {2.68297e+00, 3.04257e+00, 3.61212e+00}},
           {{3.73186e+02, 6.54084e+01, 2.40530e+00},
            {1.80541e-15, 1.37963e-17, 1.13734e-18},
            {5.70400e-01, 6.88725e+00, 4.87809e+00}},
           {{3.97823e+02, 8.33079e+01, 2.86948e+00},
            {9.32266e-16, 1.96187e-17, 1.58963e-18},
            {1.58852e+00, 4.78566e+00, 3.46207e+00}},
           {{4.30071e+02, 9.62889e+01, 2.63999e+00},
            {5.93459e-16, 2.93852e-17, 1.33987e-18},
            {2.59827e+00, 3.08148e+00, 3.59278e+00}},
           {{4.83293e+02, 9.01230e+01, 2.58417e+00},
            {4.93049e-16, 2.46581e-17, 1.18496e-18},
            {2.61633e+00, 3.18194e+00, 3.67584e+00}},
           {{2.14885e+00, 3.12000e+02, 3.10851e+02},
            {5.03211e-19, 4.49944e-16, 2.44536e-16},
            {8.07174e+00, 6.96292e+00, 7.52120e-01}},
           {{3.35553e+02, 3.38000e+02, 2.44683e+00},
            {2.38223e-16, 3.80311e-16, 6.69957e-19},
            {7.67380e-01, 7.38322e+00, 6.00575e+00}},
           {{5.05422e+02, 1.81943e+02, 4.16348e+01},
            {4.51810e-16, 1.62925e-15, 2.56670e-17},
            {9.72720e-01, 2.31429e+00, 1.05428e+00}},
           {{6.44739e+02, 9.43868e+01, 4.48739e+01},
            {5.34468e-16, 7.52449e-16, 2.62482e-17},
            {9.10320e-01, 1.21060e+01, 1.00415e+00}},
           {{7.37017e+02, 5.45084e+01, 4.94746e+01},
            {5.16504e-16, 9.45445e-16, 2.55816e-17},
            {9.83800e-01, 1.20857e+01, 1.02048e+00}},
           {{7.07575e+02, 1.32819e+02, 5.96053e+01},
            {3.06410e-16, 5.47652e-16, 2.65740e-17},
            {1.64286e+00, 1.21053e+01, 9.84130e-01}},
           {{3.80940e+00, 4.80000e+02, 4.77191e+02},
            {1.24646e-18, 6.89379e-16, 2.26522e-16},
            {3.16296e+00, 2.54708e+00, 7.17250e-01}},
           {{5.05957e+02, 5.12000e+02, 6.04261e+00},
            {2.13805e-16, 1.37078e-15, 2.17703e-18},
            {8.01490e-01, 1.08567e+00, 2.18743e+00}},
           {{4.10347e+00, 5.44000e+02, 5.40897e+02},
            {1.21448e-18, 1.22209e-15, 2.07434e-16},
            {2.94257e+00, 1.38131e+00, 7.42310e-01}},
           {{5.74665e+02, 5.78000e+02, 3.33531e+00},
            {2.02122e-16, 1.13856e-15, 8.87170e-19},
            {7.27310e-01, 1.58162e+00, 3.40997e+00}},
           {{1.55277e+01, 5.97472e+02, 6.12000e+02},
            {5.91556e-18, 9.06914e-16, 1.75583e-16},
            {1.27523e+00, 2.19900e+00, 1.01626e+00}},
           {{1.00991e+01, 6.47993e+02, 6.37908e+02},
            {3.46090e-18, 8.77868e-16, 1.81312e-16},
            {1.57383e+00, 2.20493e+00, 8.62200e-01}},
           {{4.95013e+00, 6.82009e+02, 6.82041e+02},
            {1.39331e-18, 9.70871e-16, 1.83716e-16},
            {2.33453e+00, 2.19799e+00, 7.54400e-01}},
           {{1.63391e+01, 7.22000e+02, 7.05661e+02},
            {5.47242e-18, 1.85320e-16, 2.58371e-15},
            {1.23610e+00, 8.86970e-01, 6.03320e-01}},
           {{6.20836e+00, 7.54885e+02, 7.59906e+02},
            {1.71017e-18, 1.69254e-16, 1.74416e-15},
            {1.94376e+00, 8.03230e-01, 1.23338e+00}},
           {{3.52767e+00, 7.99974e+02, 7.96498e+02},
            {7.92438e-19, 1.14059e-15, 1.74730e-16},
            {2.91332e+00, 2.15596e+00, 7.09320e-01}},
           {{2.77630e+00, 8.40000e+02, 8.38224e+02},
            {4.72225e-19, 7.90712e-16, 1.76817e-16},
            {4.01832e+00, 3.10675e+00, 6.72230e-01}},
           {{2.19565e+00, 8.82000e+02, 8.79804e+02},
            {2.74825e-19, 5.36611e-16, 1.74757e-16},
            {5.80160e+00, 4.68928e+00, 6.46550e-01}},
           {{1.22802e+01, 9.24000e+02, 9.12720e+02},
            {4.02137e-18, 8.27932e-16, 1.67390e-16},
            {1.19508e+00, 2.93024e+00, 7.61980e-01}},
           {{9.65741e+02, 9.68000e+02, 2.25892e+00},
            {1.66620e-16, 2.43290e-16, 2.68691e-19},
            {6.59260e-01, 1.02607e+01, 5.33416e+00}},
           {{1.01109e+03, 1.01200e+03, 1.90993e+00},
            {1.68841e-16, 5.82899e-16, 1.81380e-19},
            {6.37810e-01, 3.86595e+00, 6.92665e+00}},
           {{2.85583e+00, 1.05800e+03, 1.05514e+03},
            {4.73202e-19, 1.97595e-16, 1.60726e-16},
            {3.23097e+00, 1.11708e+01, 6.78350e-01}},
           {{3.65673e+00, 1.10400e+03, 1.10134e+03},
            {7.28319e-19, 1.96263e-16, 1.59441e-16},
            {2.43990e+00, 1.12867e+01, 6.74080e-01}},
           {{2.25777e+02, 1.15195e+03, 9.26275e+02},
            {3.64382e-15, 1.73961e-16, 1.36927e-16},
            {1.55583e+00, 8.29496e+00, 5.58950e-01}},
           {{1.95284e+00, 1.19905e+03, 1.20000e+03},
            {1.53323e-19, 1.62174e-16, 2.70127e-16},
            {6.96814e+00, 6.12490e-01, 8.36420e+00}},
           {{1.57750e+01, 1.25000e+03, 1.23423e+03},
            {4.15409e-18, 5.31143e-16, 1.63371e-16},
            {1.06573e+00, 4.09980e+00, 6.87760e-01}},
           {{3.99006e+01, 1.30000e+03, 1.26110e+03},
            {7.91645e-18, 5.29731e-16, 1.29776e-16},
            {8.41750e-01, 4.25068e+00, 1.02167e+00}},
           {{3.79270e+00, 1.35200e+03, 1.34821e+03},
            {6.54036e-19, 4.19760e-16, 1.49012e-16},
            {2.23516e+00, 5.67673e+00, 6.54010e-01}},
           {{6.47339e+01, 1.40400e+03, 1.34027e+03},
            {1.04123e-17, 4.91842e-16, 1.17301e-16},
            {7.91290e-01, 4.82498e+00, 1.20616e+00}},
           {{1.32391e+03, 1.45800e+03, 1.34085e+02},
            {9.11600e-17, 4.67937e-16, 1.67919e-17},
            {1.90259e+00, 5.12968e+00, 7.64980e-01}},
           {{3.73723e+00, 1.51200e+03, 1.50926e+03},
            {5.97268e-19, 4.32264e-16, 1.47596e-16},
            {2.18266e+00, 5.94532e+00, 6.30640e-01}},
           {{2.40454e+03, 7.29852e+02, 1.60851e+00},
            {1.23272e-15, 6.91046e-17, 1.14246e-19},
            {5.13050e-01, 7.16220e-01, 6.13771e+00}},
           {{2.83408e+01, 1.59666e+03, 1.62400e+03},
            {5.83259e-18, 1.62962e-16, 1.10392e-15},
            {8.83610e-01, 6.30280e-01, 2.17033e+00}},
           {{2.99869e+01, 1.68200e+03, 1.65201e+03},
            {5.42458e-18, 9.87241e-16, 1.58755e-16},
            {9.19250e-01, 2.34945e+00, 6.52360e-01}},
           {{2.17128e+02, 1.74000e+03, 1.52387e+03},
            {2.20137e-17, 1.04526e-15, 1.11706e-16},
            {6.80330e-01, 1.84671e+00, 1.66943e+00}},
           {{7.17138e+01, 1.80000e+03, 1.72829e+03},
            {1.19654e-17, 1.05819e-15, 1.80135e-16},
            {7.20780e-01, 1.66325e+00, 6.27030e-01}},
           {{2.55420e+02, 1.60579e+03, 1.85979e+03},
            {2.34810e-17, 1.10579e-16, 1.00213e-15},
            {6.62460e-01, 1.73395e+00, 1.72469e+00}},
           {{1.34495e+02, 1.78751e+03, 1.92200e+03},
            {1.53337e-17, 1.49116e-16, 9.44133e-16},
            {6.66760e-01, 9.37150e-01, 1.73686e+00}},
           {{3.36459e+03, 6.03151e+02, 1.25916e+00},
            {8.38225e-16, 4.61021e-17, 4.72200e-20},
            {4.93940e-01, 7.24970e-01, 9.86000e+00}},
           {{4.25326e+02, 2.04800e+03, 1.62267e+03},
            {3.40248e-17, 1.51430e-16, 1.18997e-15},
            {5.89240e-01, 1.74504e+00, 1.76759e+00}},
           {{4.49405e+02, 2.11200e+03, 1.66360e+03},
            {3.50901e-17, 1.53667e-16, 1.16311e-15},
            {5.75580e-01, 1.71531e+00, 1.69728e+00}},
           {{1.84046e+02, 1.99395e+03, 2.17800e+03},
            {1.95115e-17, 1.67844e-15, 2.31716e-16},
            {6.33070e-01, 5.20390e-01, 6.24360e-01}},
           {{3.10904e+03, 3.34907e+02, 1.04505e+03},
            {2.91803e-16, 2.74940e-17, 1.86238e-15},
            {8.44470e-01, 5.81910e-01, 1.76662e+00}},
           {{1.93133e+02, 2.31200e+03, 2.11887e+03},
            {1.98684e-17, 2.31253e-16, 1.53632e-15},
            {6.02960e-01, 6.14440e-01, 4.85140e-01}},
           {{3.60848e+03, 8.85149e+02, 2.67371e+02},
            {3.59425e-16, 2.27211e-15, 2.45853e-17},
            {5.67190e-01, 1.67701e+00, 5.73420e-01}},
           {{1.52967e+02, 2.33719e+03, 2.40984e+03},
            {1.54000e-17, 1.33401e-15, 2.08069e-16},
            {6.21660e-01, 5.13690e-01, 6.15180e-01}},
           {{4.84517e+02, 2.03648e+03, 2.52000e+03},
            {3.04174e-17, 9.02548e-16, 1.08659e-16},
            {5.75300e-01, 1.60766e+00, 2.18455e+00}},
           {{4.22591e+02, 2.16941e+03, 2.59200e+03},
            {2.71295e-17, 1.77743e-15, 1.29019e-16},
            {5.73290e-01, 4.66080e-01, 1.73467e+00}},
           {{4.23518e+02, 2.24149e+03, 2.66400e+03},
            {2.68030e-17, 1.76608e-15, 1.24987e-16},
            {5.58000e-01, 4.97920e-01, 1.72521e+00}},
           {{3.93404e+02, 2.34460e+03, 2.73800e+03},
            {2.36469e-17, 9.45054e-16, 1.07865e-16},
            {5.75670e-01, 1.49166e+00, 1.78600e+00}},
           {{4.37172e+02, 2.81200e+03, 2.37583e+03},
            {2.56818e-17, 1.06805e-16, 1.03501e-15},
            {5.56120e-01, 1.84906e+00, 1.35611e+00}},
           {{4.32356e+02, 2.88800e+03, 2.45564e+03},
            {2.50364e-17, 1.06085e-16, 1.05211e-15},
            {5.46070e-01, 1.80604e+00, 1.31574e+00}},
           {{4.78710e+02, 2.96400e+03, 2.48629e+03},
            {2.68180e-17, 1.01688e-16, 9.38473e-16},
            {5.32510e-01, 1.92788e+00, 1.57870e+00}},
           {{4.55097e+02, 2.91804e+03, 2.71086e+03},
            {2.56229e-17, 1.02260e-16, 8.66912e-16},
            {5.19280e-01, 1.76411e+00, 1.46877e+00}},
           {{4.95237e+02, 2.88297e+03, 2.86279e+03},
            {2.74190e-17, 7.77930e-16, 9.37780e-17},
            {5.02650e-01, 1.59305e+00, 1.89052e+00}},
           {{4.17800e+02, 2.93874e+03, 3.04346e+03},
            {2.27442e-17, 8.01660e-16, 9.91467e-17},
            {5.24450e-01, 1.58550e+00, 1.64780e+00}},
           {{3.36795e+03, 2.71613e+03, 4.76925e+02},
            {1.38078e-15, 9.18595e-17, 2.58481e-17},
            {4.92900e-01, 1.80503e+00, 5.04190e-01}},
           {{3.28171e+03, 5.11660e+02, 2.93063e+03},
            {1.49595e-15, 2.73428e-17, 9.72329e-17},
            {5.10980e-01, 4.86600e-01, 1.73998e+00}},
           {{3.61256e+03, 5.81475e+02, 2.69496e+03},
            {1.20023e-16, 3.01222e-17, 9.77921e-16},
            {1.52959e+00, 4.66490e-01, 1.79809e+00}},
           {{3.36873e+03, 5.94305e+02, 3.09296e+03},
            {1.74446e-15, 3.09814e-17, 1.02928e-16},
            {4.23340e-01, 4.55950e-01, 1.66207e+00}},
           {{3.40746e+03, 6.72232e+02, 3.14531e+03},
            {1.82836e-15, 3.39028e-17, 1.01767e-16},
            {4.12920e-01, 4.43740e-01, 1.73089e+00}},
           {{4.02866e+01, 3.65771e+03, 3.69800e+03},
            {5.80108e-18, 1.49653e-15, 1.81276e-16},
            {1.01250e+00, 5.48650e-01, 3.48350e-01}},
           {{6.41240e+02, 3.14376e+03, 3.78400e+03},
            {3.02324e-17, 1.19511e-15, 1.07026e-16},
            {4.50150e-01, 1.45661e+00, 1.59656e+00}},
           {{8.26440e+02, 3.04556e+03, 3.87200e+03},
            {3.71029e-17, 1.40408e-15, 1.11273e-16},
            {4.30670e-01, 1.43268e+00, 1.70060e+00}},
           {{3.57913e+03, 3.66670e+03, 6.75166e+02},
            {1.01058e-16, 2.37226e-15, 3.25695e-17},
            {1.60260e+00, 3.53520e-01, 4.18670e-01}},
           {{4.91644e+03, 1.59784e+03, 1.58571e+03},
            {4.87707e-16, 8.35973e-17, 1.77629e-15},
            {3.92610e-01, 3.59110e-01, 3.26255e+00}},
           {{9.30184e+02, 3.42887e+03, 3.92195e+03},
            {4.18953e-17, 1.40890e-15, 1.18382e-16},
            {3.85590e-01, 1.26339e+00, 1.47985e+00}},
           {{8.87945e+02, 3.68122e+03, 3.89483e+03},
            {4.03182e-17, 1.28190e-15, 1.11100e-16},
            {3.75750e-01, 1.26838e+00, 1.47126e+00}},
           {{3.49096e+03, 1.14331e+03, 4.01473e+03},
            {1.11553e-16, 4.96925e-17, 1.56996e-15},
            {1.53155e+00, 3.58770e-01, 7.25730e-01}},
           {{4.05860e+03, 1.64717e+03, 3.13023e+03},
            {9.51125e-16, 6.04886e-17, 8.45221e-17},
            {1.51924e+00, 3.78260e-01, 2.44856e+00}},
           {{3.06810e+03, 1.44490e+03, 4.51200e+03},
            {2.57569e-15, 7.39507e-17, 3.67830e-16},
            {3.23860e-01, 3.49900e-01, 3.64510e-01}},
           {{3.89832e+03, 1.89433e+03, 3.42335e+03},
            {1.14294e-15, 6.68320e-17, 1.20652e-16},
            {1.31791e+00, 3.65740e-01, 1.87150e+00}},
           {{1.39834e+03, 3.30912e+03, 4.70153e+03},
            {2.98597e-15, 1.09433e-16, 3.91104e-16},
            {1.47722e+00, 3.36540e-01, 1.35731e+00}},
           {{5.28518e+03, 2.33859e+03, 1.98023e+03},
            {5.88714e-16, 9.61804e-17, 3.52282e-15},
            {3.35840e-01, 3.30010e-01, 2.81960e-01}},
           {{1.00000e+00, 4.90000e+03, 4.90000e+03},
            {1.46196e-20, 1.38525e-16, 4.29979e-16},
            {8.60979e+00, 3.76480e-01, 3.12240e+00}},
           {{8.72368e+02, 4.85661e+03, 4.27102e+03},
            {1.53226e-15, 2.49104e-16, 1.28308e-16},
            {5.84949e+00, 3.28173e+00, 3.26330e-01}}};

    CELER_VALIDATE((z - 1) * sizeof(ElScatParams) < sizeof(el_params),
                   << "atomic number " << z
                   << " is out of range for Rayleigh model data (must be less "
                      "than "
                   << sizeof(el_params) / sizeof(ElScatParams) << ")");
    return el_params[z - 1];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
