//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file testdetail/JsonComparer.json.cc
//---------------------------------------------------------------------------//
#include "JsonComparer.hh"

#include <nlohmann/json.hpp>

using nlohmann::json;

namespace celeritas
{
namespace testdetail
{
namespace
{
//---------------------------------------------------------------------------//
void convert(char const* label,
             std::string_view s,
             json* result,
             ::testing::AssertionResult* failure)
{
    if (s.empty())
    {
        *failure = ::testing::AssertionFailure();
        (*failure) << label << " is an empty string";
        return;
    }

    try
    {
        *result = json::parse(s.begin(), s.end());
    }
    catch (json::parse_error const& j)
    {
        *failure = ::testing::AssertionFailure();
        (*failure) << "Failed to parse " << label << ": " << j.what();
    }
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Implementation class for comparison.
 */
struct JsonComparer::Impl
{
    JsonComparer::Compare const& soft_eq;
    JsonComparer::VecFailure* failures{nullptr};
    std::vector<std::string> key_stack;

    // Recursively test for equality
    void operator()(json& expected, json& actual);

    void add_failure(std::string&& what,
                     std::string&& expected,
                     std::string&& actual) const;
};

//---------------------------------------------------------------------------//
auto JsonComparer::operator()(std::string_view expected,
                              std::string_view actual) -> result_type
{
    ::testing::AssertionResult result = ::testing::AssertionSuccess();
    json exp_j;
    convert("expected", expected, &exp_j, &result);
    if (!result)
    {
        return result;
    }
    json act_j;
    convert("actual", actual, &act_j, &result);
    if (!result)
    {
        return result;
    }

    VecFailure failures;
    Impl compare_impl{compare_, &failures, {"."}};
    compare_impl(exp_j, act_j);

    if (!failures.empty())
    {
        result = ::testing::AssertionFailure();
        result << "JSON objects differ:";
        for (auto const& f : failures)
        {
            result << "\n  ";
            result << f.what << " in " << f.where << ": expected "
                   << f.expected;
            if (!f.actual.empty())
            {
                result << ", but got " << f.actual;
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
void JsonComparer::Impl::operator()(json& a, json& b)
{
    using std::to_string;

    if (a.type() != b.type())
    {
        this->add_failure("type", a.type_name(), b.type_name());
    }
    else if (a.size() != b.size())
    {
        this->add_failure("size", to_string(a.size()), to_string(b.size()));
        // TODO: for objects, print set operation on keys?
    }
    else if (a.is_object())
    {
        for (auto const& [key, a_value] : a.items())
        {
            auto b_iter = b.find(key);
            if (b_iter == b.end())
            {
                this->add_failure("missing key", std::string(key), {});
                continue;
            }
            this->key_stack.push_back("[\"" + key + "\"]");
            (*this)(a_value, *b_iter);
            this->key_stack.pop_back();
        }
    }
    else if (a.is_array())
    {
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            this->key_stack.push_back("[" + std::to_string(i) + "]");
            (*this)(a[i], b[i]);
            this->key_stack.pop_back();
        }
    }
    else if (a.is_number_float())
    {
        // Compare with "native" tolerance
        // using FloatT = json::number_float_t;
        if (!this->soft_eq(a.get<real_type>(), b.get<real_type>()))
        {
            this->add_failure("value", a.dump(), b.dump());
        }
    }
    else if (a != b)
    {
        this->add_failure("value", a.dump(), b.dump());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Push a failure onto the stack.
 */
void JsonComparer::Impl::add_failure(std::string&& what,
                                     std::string&& expected,
                                     std::string&& actual) const
{
    Failure f;
    for (auto const& s : key_stack)
    {
        f.where += s;
    }
    f.what = std::move(what);
    f.expected = std::move(expected);
    f.actual = std::move(actual);
    failures->push_back(std::move(f));
}

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas
