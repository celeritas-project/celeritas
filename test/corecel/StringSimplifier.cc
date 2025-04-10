//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/StringSimplifier.cc
//---------------------------------------------------------------------------//
#include "StringSimplifier.hh"

#include <iomanip>
#include <regex>
#include <sstream>

namespace celeritas
{
namespace test
{
namespace
{
void erase_exp_zeros(std::string& s)
{
    // If using scientific notation, simplify the exponent
    auto exp_iter = std::find_if(
        s.begin(), s.end(), [](char c) { return c == 'e' || c == 'E'; });
    CELER_ASSERT(exp_iter != s.end());
    ++exp_iter;
    CELER_ASSERT(exp_iter != s.end());
    // Start of the exponent number, likely with leading -/+
    auto num_iter = exp_iter;
    if (*num_iter == '-')
    {
        // Don't erase the -
        ++num_iter;
        ++exp_iter;
    }
    else if (*num_iter == '+')
    {
        ++num_iter;
    }
    // Search up to the last digit
    CELER_ASSERT(num_iter != s.end());
    auto end_iter
        = std::find_if(num_iter, s.end() - 1, [](char c) { return c != '0'; });
    s.erase(exp_iter, end_iter);
}

};  // namespace
//---------------------------------------------------------------------------//
/*!
 * Simplify.
 */
[[nodiscard]] std::string
StringSimplifier::operator()(std::string const& old) const
{
    // Regex to match floating point numbers (both standard and scientific
    // notation) with optional scientific notation and trailing f
    static std::regex const combined_regex(
        "(?:"
        R"(((?:-?\d*\.\d+|-?\d+\.\d*)(?:[eE][-+]?\d+)?f?))"
        "|"
        "(\033\\[[0-9;]*m)"  // ANSI
        "|"
        "(0x[0-9a-f]+)"  // Pointer
        ")");

    std::string result;
    result.reserve(old.size());
    auto current = old.cbegin();
    std::smatch match;

    // Find all matches and process them one by one
    while (std::regex_search(current, old.cend(), match, combined_regex))
    {
        // Add the text between the last match and this one
        result.append(current, match[0].first);

        if (match[1].length() > 0)
        {
            result.append(this->simplify_float(match[1].str()));
        }
        else if (match[2].length() > 0)
        {
            // Omit ANSI
        }
        else if (match[3].length() > 0)
        {
            // Add placeholder pointer
            result.append("0x0");
        }

        // Update current position
        current = match[0].second;
    }

    // Add remaining text after the last match
    result.append(current, old.cend());

    return result;
}

std::string StringSimplifier::simplify_float(std::string&& s) const
{
    CELER_EXPECT(!s.empty());

    // Position is start of the mantissa, or the end of the number
    auto end = s.cend() - 1;
    if (*end != 'f')
        ++end;

    auto iter = std::find(s.cbegin(), end, '.');
    ++iter;
    auto exp_iter
        = std::find_if(iter, end, [](auto c) { return c == 'e' || c == 'E'; });

    int const precision
        = std::min<int>(std::distance(iter, exp_iter), float_digits_);
    bool const is_scientific = (exp_iter != end);

    // Format the rounded number with appropriate notation using IIFE
    auto new_s = [precision, value = std::stod(s), is_scientific]() {
        std::ostringstream oss;
        if (is_scientific)
        {
            oss << std::scientific;
        }
        else
        {
            oss << std::fixed;
        }
        oss << std::setprecision(precision) << value;
        return std::move(oss).str();
    }();
    CELER_ASSERT(!new_s.empty());
    if (precision == 0 && !is_scientific && new_s.back() != '.')
    {
        // Add trailing decimal
        new_s.push_back('.');
    }

    if (is_scientific)
    {
        // Change e+001 -> e1
        erase_exp_zeros(new_s);
    }

    return new_s;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
