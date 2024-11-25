//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ValidationUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/io/ImportPhysicsVector.hh"
#include "celeritas/optical/MfpBuilder.hh"

#include "Test.hh"
#include "TestMacros.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//
/*!
 * Helper function to annotate units of a hard-coded test data array.
 *
 * Converts the arguments supplied in units \c UnitType to native units.
 */
template<class UnitType, class... Args>
std::array<real_type, sizeof...(Args)> constexpr expressed_as(Args const&... args)
{
    return std::array<real_type, sizeof...(Args)>{
        native_value_from(UnitType{args})...};
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to annotate units of hard-coded test data.
 *
 * Same as \c expressed_as, except returns a vector.
 */
template<class UnitType, class... Args>
std::vector<real_type> vector_expressed_as(Args const&... args)
{
    return std::vector<real_type>{native_value_from(UnitType{args})...};
}

//---------------------------------------------------------------------------//
/*!
 * Checks two \c ImportPhysicVector for exact equality.
 */
void check_physics_vector(ImportPhysicsVector const& expected,
                          ImportPhysicsVector const& actual);

//---------------------------------------------------------------------------//
/*!
 * Convert a floating point type collection into a \c real_type span.
 *
 * Used to convert data which may be expressed as double or single precision
 * into the same size as \c real_type. If the supplied data is the same
 * precision, then just a span to the original data is returned. Otherwise,
 * it is copied into a local buffer and casted to the correct precision. The
 * \c RealSpanGenerator object should live as long as its spans are being
 * used in the tests.
 */
class RealSpanGenerator
{
  public:
    /*!
     * Create a \c real_type \c Span for the supplied collection type \c T.
     *
     * If \c T::value_type is \c real_type, then a span to the original data
     * is returned. Otherwise, it is copied to a local buffer and cast to
     * \c real_type.
     */
    template<class T>
    Span<real_type const> operator()(T const& xs)
    {
        if constexpr (std::is_same_v<real_type,
                                     std::remove_cv_t<typename T::value_type>>)
        {
            return make_span(xs);
        }
        else
        {
            buffer_.resize(xs.size());
            for (auto i : range(xs.size()))
            {
                buffer_[i] = static_cast<real_type>(xs[i]);
            }
            return make_span(buffer_);
        }
    }

  private:
    std::vector<real_type> buffer_;
};

//---------------------------------------------------------------------------//
/*!
 * Perform consistency checks on grids built in \c Collections.
 */
class GridValidator
{
  public:
    //!@{
    //! \name Type aliases
    using Grid = GenericGridRecord;
    using GridId = OpaqueId<Grid>;
    using ImportPhysicsTable = std::vector<ImportPhysicsVector>;

    template<class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;
    //!@}

    //! Whether to use soft or exact equivalence
    enum Softness
    {
        Soft,
        Exact
    };

  public:
    // Construct validator for underlying storage
    GridValidator(Items<real_type>* reals, Items<Grid>* grids);

    // Check the imported data is built under the given grid ID range
    void check_built_table(ImportPhysicsTable const& table,
                           ItemRange<Grid> grid_ids,
                           Softness soft);

    // Check the imported data is built under the given grid ID
    void check_built_grid(ImportPhysicsVector const& expected,
                          GridId grid_id,
                          Softness soft);

    //! Check the imported data is built in the given data range
    template<class T>
    void check_built_vector(T const& t,
                            ItemRange<real_type> const& real_ids,
                            Softness soft)
    {
        this->check_span(convert_real_(t), real_ids, soft);
    }

    // Construct an MFP builder with the underlying collections
    MfpBuilder create_mfp_builder();

  private:
    Items<real_type>* reals_;
    Items<Grid>* grids_;
    RealSpanGenerator convert_real_;

    // Check the imported data is built in the given data range
    void check_span(Span<real_type const> const& t,
                    ItemRange<real_type> const& real_ids,
                    Softness soft);
};

//---------------------------------------------------------------------------//
/*!
 * A \c GridValidator that stores its own collections.
 */
class GridStorage : public GridValidator
{
  public:
    // Construct with internal collections
    GridStorage();

  private:
    Items<real_type> reals_;
    Items<Grid> grids_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
