//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/msc/MscStepToGeo.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <iostream>

#include "corecel/io/Repr.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"

#include "UrbanMscHelper.hh"
using std::cout;
using std::endl;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Convert the "true" path traveled to a geometrical approximatio.
 *
 * This takes the physical step limit---whether limited by range, physics
 * interaction, or an MSC step limiter---and converts it to a "geometrical"
 * path length, which is a smooth curve (straight if in the absence of a
 * magnetic field).
 *
 * The mean value of the geometrical path length \f$ z \f$ (the first moment)
 * corresponding to a given true path length \f$ t \f$ is given by
 * \f[
 *     \langle z \rangle = \lambda_{1} [ 1 - \exp({-\frac{t}{\lambda_{1}}})]
 * \f]
 * where \f$\lambda_{1}\f$ is the first transport mean free path. Due to the
 * fact that \f$\lambda_{1}\f$ depends on the kinetic energy of the path and
 * decreases along the step, the path length correction is approximated as
 * \f[
 *     \lambda_{1} (t) = \lambda_{10} (1 - \alpha t)
 * \f]
 * where \f$ \alpha = \frac{\lambda_{10} - \lambda_{11}}{t\lambda_{10}} \f$
 * or  \f$ \alpha = 1/r_0 \f$ in a simpler form with the range \f$ r_0 \f$
 * if the kinetic energy of the particle is below its mass -
 * \f$ \lambda_{10} (\lambda_{11}) \f$ denotes the value of \f$\lambda_{1}\f$
 * at the start (end) of the step, respectively.
 *
 * Since the MSC cross section decreases as the energy increases, \f$
 * \lambda_{10} \f$ will be larger than \f$ \lambda_{11} \f$ and \f$ \alpha \f$
 * will be positive. However, in the Geant4 Urban MSC model, different methods
 * are used to calculate the cross section above and below 10 MeV. In the
 * higher energy region the cross sections are identical for electrons and
 * positrons, resulting in a discontinuity in the positron cross section at 10
 * MeV. This means on fine energy grids it's possible for the cross section to
 * be *increasing* with energy just above the 10 MeV threshold and therefore
 * for \f$ \alpha \f$ is negative.
 *
 * \note This performs the same method as in ComputeGeomPathLength of
 * G4UrbanMscModel of the Geant4 10.7 release.
 */
class MscStepToGeo
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using UrbanMscRef = NativeCRef<UrbanMscData>;
    //!@}

    struct result_type
    {
        real_type step{};  //!< Geometrical step length
        real_type alpha{0};  //!< Scaled MFP slope
    };

  public:
    // Construct from MSC data
    inline MscStepToGeo(UrbanMscRef const& shared,
                        UrbanMscHelper const& helper,
                        Energy energy,
                        real_type lambda,
                        real_type range);

    // Calculate the geometrical step
    inline CELER_FUNCTION result_type operator()(real_type tstep) const;

  private:
    UrbanMscRef const& shared_;
    UrbanMscHelper const& helper_;
    real_type energy_;
    real_type lambda_;
    real_type range_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from MSC data.
 */
CELER_FUNCTION MscStepToGeo::MscStepToGeo(UrbanMscRef const& shared,
                                          UrbanMscHelper const& helper,
                                          Energy energy,
                                          real_type lambda,
                                          real_type range)
    : shared_(shared)
    , helper_(helper)
    , energy_(energy.value())
    , lambda_(lambda)
    , range_(range)
{
    CELER_EXPECT(energy_ > 0);
    CELER_EXPECT(lambda_ > 0);
    CELER_EXPECT(range_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the geometry step length for a given true step length.
 */
CELER_FUNCTION auto MscStepToGeo::operator()(real_type tstep) const
    -> result_type
{
    cout << "z -> g: ";
    result_type result;
    result.alpha = MscStep::small_step_alpha();
    if (tstep < shared_.params.min_step())
    {
        // Geometrical path length = true path length for a very small step
        result.step = tstep;
        cout << "Very small step" << endl;
    }
    else if (tstep < range_ * shared_.params.dtrl())
    {
        // Small enough distance to assume cross section is constant
        // over the step: z = lambda * (1 - exp(-tau))
        result.step = -lambda_ * std::expm1(-tstep / lambda_);
        cout << "Constant cross section, tau = " << repr(tstep / lambda_)
             << endl;
    }
    else
    {
        // Eq 8.8: mfp_slope = (lambda_f / lambda_i) = (1 - alpha * tstep)
        real_type mfp_slope;
        if (energy_ < value_as<Mass>(shared_.electron_mass) || tstep == range_)
        {
            // Low energy or range-limited step
            // (For range-limited step, the final cross section and thus MFP
            // slope are zero).
            result.alpha = 1 / range_;
            mfp_slope = 1 - result.alpha * tstep;
            cout << "Low energy or range-limited step: slope ="
                 << repr(mfp_slope) << endl;
        }
        else
        {
            // Calculate the energy at the end of a physics-limited step
            real_type rfinal
                = max<real_type>(range_ - tstep, real_type(0.01) * range_);
            Energy endpoint_energy = helper_.calc_stopping_energy(rfinal);
            real_type lambda1 = helper_.calc_msc_mfp(endpoint_energy);

            // Calculate the geometric path assuming the cross section is
            // linear between the start and end energy. Eq 8.10+1
            result.alpha = (lambda_ - lambda1) / (lambda_ * tstep);
            CELER_ASSERT(result.alpha != MscStep::small_step_alpha());
            mfp_slope = lambda1 / lambda_;
            cout << "Physics-limited step: slope =" << repr(mfp_slope) << endl;
        }

        // Eq 8.10 with simplifications
        real_type w = 1 + 1 / (result.alpha * lambda_);
        result.step = (1 - fastpow(mfp_slope, w)) / (result.alpha * w);
    }

    if (result.step > lambda_)
    {
        cout << "Reducing step " << result.step << " to lambda = " << lambda_
             << endl;
    }
    // Limit step to 1 MFP
    result.step = min<real_type>(result.step, lambda_);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
