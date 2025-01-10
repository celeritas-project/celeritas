//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformVisitor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "orange/OrangeData.hh"

#include "NoTransformation.hh"
#include "TransformTypeTraits.hh"
#include "Transformation.hh"
#include "Translation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply a functor to a type-deleted transform.
 *
 * An instance of this class is like \c std::visit but accepting a \c
 * TransformId rather than a \c std::variant .
 *
 * Example: \code
 TransformVisitor visit_transform{params_};
 auto new_pos = visit_transform(
    [&pos](auto&& t) { return t.transform_up(pos); },
    daughter.trans_id);
 \endcode
 */
class TransformVisitor
{
    template<class T>
    using Items = Collection<T, Ownership::const_reference, MemSpace::native>;

  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<OrangeParamsData>;
    using TransformRecords = Items<TransformRecord>;
    using Reals = Items<real_type>;
    //!@}

  public:
    // Construct manually from required data
    inline CELER_FUNCTION
    TransformVisitor(TransformRecords const& transforms, Reals const& reals);

    // Construct from ORANGE params
    explicit inline CELER_FUNCTION TransformVisitor(ParamsRef const& params);

    // Apply the function to the transform specified by the given ID
    template<class F>
    inline CELER_FUNCTION decltype(auto)
    operator()(F&& typed_visitor, TransformId t);

  private:
    TransformRecords const& transforms_;
    Reals const& reals_;

    // Construct a transform from a data offset
    template<class T>
    inline CELER_FUNCTION T make_transform(OpaqueId<real_type> data_offset) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct manually from required data.
 */
CELER_FUNCTION
TransformVisitor::TransformVisitor(TransformRecords const& transforms,
                                   Reals const& reals)
    : transforms_{transforms}, reals_{reals}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct from ORANGE data.
 */
CELER_FUNCTION TransformVisitor::TransformVisitor(ParamsRef const& params)
    : TransformVisitor{params.transforms, params.reals}
{
}

#if !defined(__DOXYGEN__) || __DOXYGEN__ > 0x010908
//---------------------------------------------------------------------------//
/*!
 * Apply the function to the transform specified by the given ID.
 */
template<class F>
CELER_FUNCTION decltype(auto)
TransformVisitor::operator()(F&& func, TransformId id)
{
    CELER_EXPECT(id < transforms_.size());

    // Load transform record (type + data)
    TransformRecord const tr = transforms_[id];
    CELER_ASSERT(tr);

    // Apply type-deleted functor based on type
    return visit_transform_type(
        [&](auto tt_traits) {
            // Call the user-provided action using the reconstructed transform
            using TF = typename decltype(tt_traits)::type;
            return func(this->make_transform<TF>(tr.data_offset));
        },
        tr.type);
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Apply the function to the transform specified by the given ID.
 */
template<class T>
CELER_FUNCTION T
TransformVisitor::make_transform(OpaqueId<real_type> data_offset) const
{
    CELER_EXPECT(data_offset <= reals_.size());
    constexpr size_type size{T::StorageSpan::extent};
    CELER_ASSERT(data_offset + size <= reals_.size());

    return T{reals_[Reals::ItemRangeT{data_offset, data_offset + size}]};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
