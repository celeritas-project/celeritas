//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgObject.cc
//---------------------------------------------------------------------------//
#include "CsgObject.hh"

#include <utility>

#include "corecel/io/JsonPimpl.hh"

#include "ObjectIO.json.hh"

#include "detail/CsgUnitBuilder.hh"
#include "detail/VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
// NEGATED
//---------------------------------------------------------------------------//
/*!
 * Construct with the object to negate and an empty name.
 */
NegatedObject::NegatedObject(SPConstObject obj)
    : NegatedObject{{}, std::move(obj)}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a name and object.
 */
NegatedObject::NegatedObject(std::string&& label, SPConstObject obj)
    : label_{std::move(label)}, obj_{std::move(obj)}
{
    CELER_EXPECT(obj_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume from this object.
 */
NodeId NegatedObject::build(VolumeBuilder& vb) const
{
    // Build object to be negated
    auto daughter_id = obj_->build(vb);
    // Add the new region (or anti-region)
    return vb.insert_region(Label{label_}, Negated{daughter_id});
}

//---------------------------------------------------------------------------//
/*!
 * Output to JSON.
 */
void NegatedObject::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
// JOIN_OBJECTS
//---------------------------------------------------------------------------//
template<OperatorToken Op>
constexpr OperatorToken JoinObjects<Op>::op_token;

//---------------------------------------------------------------------------//
/*!
 * Construct a joined object if nontrivial, or return the original.
 *
 * This should only be called if the label of the resulting object is not
 * important.
 */
template<OperatorToken Op>
auto JoinObjects<Op>::or_object(std::string&& label,
                                VecObject&& objects) -> SPConstObject
{
    CELER_EXPECT(!objects.empty());
    if (objects.size() == 1)
    {
        SPConstObject result = std::move(objects.front());
        objects.clear();
        return result;
    }
    return std::make_shared<JoinObjects<Op>>(std::move(label),
                                             std::move(objects));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a name and a vector of objects.
 */
template<OperatorToken Op>
JoinObjects<Op>::JoinObjects(std::string&& label, VecObject&& objects)
    : label_{std::move(label)}, objects_{std::move(objects)}
{
    CELER_EXPECT(!label_.empty());
    CELER_EXPECT(std::all_of(objects_.begin(), objects_.end(), LogicalTrue{}));
    CELER_EXPECT(!objects_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume from the joined objets.
 */
template<OperatorToken Op>
NodeId JoinObjects<Op>::build(VolumeBuilder& vb) const
{
    // Vector of nodes and cumulative bounding zone being built
    std::vector<NodeId> nodes;
    for (auto const& obj : objects_)
    {
        // Construct the daughter CSG node
        auto daughter_id = obj->build(vb);
        nodes.push_back(daughter_id);
    }

    // Add the combined region
    return vb.insert_region(Label{label_}, Joined{op_token, std::move(nodes)});
}

//---------------------------------------------------------------------------//
/*!
 * Output to JSON.
 */
template<OperatorToken Op>
void JoinObjects<Op>::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Make a new object that is the second object subtracted from the first.
 *
 * This just takes the intersection the first object and the negated second:
 * \verbatim A - B <=> A & ~B \endverbatim
 */
std::shared_ptr<AllObjects const>
make_subtraction(std::string&& label,
                 SPConstObject const& minuend,
                 SPConstObject const& subtrahend)
{
    CELER_EXPECT(!label.empty());
    CELER_EXPECT(minuend && subtrahend);

    return std::make_shared<AllObjects>(
        std::move(label),
        AllObjects::VecObject{
            {minuend, std::make_shared<NegatedObject>(subtrahend)}});
}

//---------------------------------------------------------------------------//
/*!
 * Make a combination of possibly negated objects.
 *
 * The Region Definition Vector is the SCALE way for defining media,
 * boundaries, etc. It must not be empty.
 */
std::shared_ptr<AllObjects const>
make_rdv(std::string&& label, VecSenseObj&& inp)
{
    CELER_EXPECT(!label.empty());
    CELER_EXPECT(!inp.empty());

    AllObjects::VecObject objects;
    for (auto&& [sense, obj] : std::move(inp))
    {
        CELER_EXPECT(obj);
        // Negate 'outside' objects
        if (sense == Sense::outside)
        {
            obj = std::make_shared<NegatedObject>(obj);
        }
        objects.push_back(std::move(obj));
    }

    return std::make_shared<AllObjects>(std::move(label), std::move(objects));
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class JoinObjects<op_and>;
template class JoinObjects<op_or>;

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
