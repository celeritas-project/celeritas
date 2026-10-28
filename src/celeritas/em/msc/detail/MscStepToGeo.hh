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
 *     \langle z \rangle = \lambda_{1} [ 1 - \exp({-\frac{t}{\lambda_{1}}})] \,,
 * \f]
 * where \f$\lambda_{1}\f$ is the first transport mean free path.
 *
 * Since \f$\lambda_{1}\f$ depends on the kinetic energy of the path and
 * decreases along the step, the path length correction can be approximated as a
 * linear function over the step:
 * \f[
 *   \lambda_{1} (E(t')) \approx \lambda_{1,i} (1 - \alpha t') \,, 0 \le t' \le t
 * \f]
 * using the tabulated MFP with the start-of-step energy
 * \f$ \lambda_{1,i} \equiv \lambda_{1}(E_0)\f$,.
 * In the usual case, the linearity coefficient \f$ \alpha\f$ is calculated
 * using the calculated mean free paths from the energy at the beginning and
 * estimated step endpoint energy:
 * \f[
    \alpha \equiv \frac{\lambda_{1,i} - \lambda_{1,f}}{t \lambda_{1,i}} \,,
 * \f]
 * where
 * \f$ \lambda_{1,f} \equiv \lambda_{1}(E(t)) \f$.
 * The following edge cases calculate alpha differently:
 * - For very small steps (less than \c
 *   celeritas::UrbanMscParameters::min_step_transform), the cross
 *   section is assumed constant due to the small energy loss:
 *   \f$ \alpha = 0 \f$ . The step is not transformed.
 * - For small steps relative to the range, the cross section is approximated
 *   as constant, but the step is transformed analytically.
 * - For range-limited steps (\f$ t = r = \mathrm{range} \f$),
 *   \f$ \alpha = 1/r \f$ by definition since the mean free path at the end
 *   point of the step energy) is zero.
 * - For other low-energy steps (kinetic energy less than mass energy),
 *   \f$ \alpha = 1/r \f$ rather than using the actual endpoint cross section.
 *   This is equivalent to approximating the endpoint MFP somewhere
 *   between the initial MFP (as with the small-step case) and zero (the
 *   range-limited case).
 *
 * Since the MSC cross section should increase with decreasing energy,
 * \f$ \lambda_{1,f} \f$ will be smaller than \f$ \lambda_{1,i} \f$ and
 * \f$ \alpha \f$ will be positive.
 * However, in the Geant4 Urban MSC model, different methods
 * are used to calculate the cross section above and below 10 MeV.
 * In the higher energy region the cross sections are identical for electrons and
 * positrons, resulting in a discontinuity in the positron cross section at 10
 * MeV.
 * This means on fine energy grids it's possible for the cross section to
 * be *increasing* with energy just above the 10 MeV threshold and therefore
 * for \f$ \alpha \f$ is negative.
 *
 * The resulting geometrical step length \em can be greater than 1 MFP: it's the
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
CELER_FUNCTION auto MscStepToGeo::operator()(real_type tstep) const
    -> result_type
{
    using namespace celeritas::literals;
    CELER_EXPECT(tstep >= 0_r && tstep <= range_);

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
        // Calculate end-of-step MSC MFP
        real_type lambda_f = [this, tstep] {
            if (tstep == range_)
            {
                // MFP at rest (endpoint) is zero
                return 0_r;
            }
            if (energy_ < value_as<Mass>(shared_.electron_mass))
            {
                // Low-energy step: set alpha = 1 / range
                // Equivalently: the small-step case has no change in xs,
                // and the large-step case has final lambda of zero
                return lambda_ * (1_r - tstep / range_);
            }

            // Calculate the energy at the end of the physics-limited step
            Energy endpoint_energy = helper_.calc_inverse_range(range_ - tstep);
            return helper_.calc_msc_mfp(endpoint_energy);
        }();

        // Eq 8.8: lambda_frac = (lambda_f / lambda_i) = (1 - alpha * tstep)
        // Calculate the geometric path assuming the cross section is
        // linear between the start and end energy. Eq 8.10+1
        real_type lambda_frac = lambda_f / lambda_;
        result.alpha = (1_r - lambda_frac) / tstep;
        // The MFP *cannot* be zero (which will lead to a NaN step);
        // the earlier exceptions should prevent catastrophic errors
        CELER_ASSERT(result.alpha != 0_r);

        // Eq 8.10 with simplifications
        real_type w = 1_r + 1_r / (result.alpha * lambda_);
        result.step = (1_r - fastpow(lambda_frac, w)) / (result.alpha * w);
    }

    // For extremely large lambda we can slightly exceed the step
    CELER_ENSURE(result.step <= tstep || soft_equal(result.step, tstep));
    result.step = min(result.step, tstep);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
