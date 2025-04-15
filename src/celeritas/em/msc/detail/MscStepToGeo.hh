//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/msc/detail/MscStepToGeo.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/math/Algorithms.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/phys/Interaction.hh"

#include "UrbanMscHelper.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Convert the "true" path traveled to a geometrical approximation.
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
 *     \lambda_{1} (t') = \lambda_{1,i} (1 - \alpha t') \,, 0 \le t' \le t
 * \f]
 * where the transport MFP at the start of the step is \f$ \lambda_{1,i} \def
 * \lambda_{1}(0)\f$ , and a linearity coefficient is defined as \f$ \alpha
 * \def \frac{\lambda_{1,i} - \lambda_{1,f}}{t \lambda_{1,i}} \f$, where  \f$
 * \lambda_{1,f}  \def \lambda_{1}(t) \f$. For low-energy or range-limited
 * steps, \f$ \alpha = 1/r_0 \f$ because the cross section at zero energy is
 * effectively infinite.
 *
 * Since the MSC cross section generally decreases as the energy increases,
 * \f$ \lambda_{1,i} \f$ will be larger than \f$ \lambda_{1,f} \f$ and
 * \f$ \alpha \f$ will be positive.
 * However, in the Geant4 Urban MSC model, different methods
 * are used to calculate the cross section above and below 10 MeV. In the
 * higher energy region the cross sections are identical for electrons and
 * positrons, resulting in a discontinuity in the positron cross section at 10
 * MeV. This means on fine energy grids it's possible for the cross section to
 * be *increasing* with energy just above the 10 MeV threshold and therefore
 * for \f$ \alpha \f$ is negative.
 *
 * The resulting geometrical step length *can* be greater than 1 MFP: it's the
 * MSC step limiter's job to add additional restrictions.
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
        real_type step{};  //!< Geometrical step length [len]
        real_type alpha{0};  //!< Scaled MFP slope [1/len]
    };

  public:
    // Construct from MSC data
    inline CELER_FUNCTION MscStepToGeo(UrbanMscRef const& shared,
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
CELER_FUNCTION auto
MscStepToGeo::operator()(real_type tstep) const -> result_type
{
    CELER_EXPECT(tstep >= 0 && tstep <= range_);

    result_type result;
    result.alpha = MscStep::small_step_alpha();
    if (tstep < shared_.params.min_step_transform)
    {
        // Very small step: do not transform step length
        result.step = tstep;
    }
    else if (tstep < range_ * shared_.params.small_range_frac)
    {
        // Small enough distance to assume cross section is constant
        // over the step: z = lambda * (1 - exp(-tau))
        result.step = -lambda_ * std::expm1(-tstep / lambda_);
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
            // Use max to avoid slightly negative slope due to roundoff for
            // range-limited steps
            mfp_slope = max<real_type>(1 - result.alpha * tstep, 0);
        }
        else
        {
            // Calculate the energy at the end of a physics-limited step
            real_type rfinal = range_ - tstep;
            Energy endpoint_energy = helper_.calc_inverse_range(rfinal);
            real_type lambda1 = helper_.calc_msc_mfp(endpoint_energy);

            // Calculate the geometric path assuming the cross section is
            // linear between the start and end energy. Eq 8.10+1
            result.alpha = (lambda_ - lambda1) / (lambda_ * tstep);
            mfp_slope = lambda1 / lambda_;
        }

        // Eq 8.10 with simplifications
        real_type w = 1 + 1 / (result.alpha * lambda_);
        result.step = (1 - fastpow(mfp_slope, w)) / (result.alpha * w);
    }

    // For extremely large lambda we can slightly exceed the step
    CELER_ENSURE(result.step <= tstep || soft_equal(result.step, tstep));
    result.step = min(result.step, tstep);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
