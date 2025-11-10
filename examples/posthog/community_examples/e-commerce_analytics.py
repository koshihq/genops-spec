#!/usr/bin/env python3
"""
E-Commerce Analytics with PostHog + GenOps

This example demonstrates comprehensive e-commerce analytics tracking with PostHog
and GenOps governance. It shows how to track user journeys, product interactions,
conversions, and revenue while maintaining cost intelligence and team attribution.

Use Case:
    - Online retail store tracking user behavior
    - Product catalog and search analytics
    - Shopping cart and checkout flow monitoring
    - Revenue and conversion tracking with governance

Usage:
    python community_examples/e-commerce_analytics.py

Prerequisites:
    pip install genops[posthog]
    export POSTHOG_API_KEY="phc_your_project_api_key"
    export GENOPS_TEAM="ecommerce-team"
    export GENOPS_PROJECT="online-store-analytics"

Expected Output:
    Complete e-commerce user journey tracking with detailed cost attribution,
    conversion funnel analysis, and revenue metrics with governance.

Learning Objectives:
    - E-commerce event taxonomy and tracking patterns
    - Revenue attribution and conversion funnel analysis
    - Cost-optimized high-volume event tracking
    - Customer lifecycle and retention analytics

Author: GenOps AI Community
License: Apache 2.0
"""

import random
import time


def main():
    """Demonstrate comprehensive e-commerce analytics with PostHog + GenOps."""
    print("🛒 E-Commerce Analytics with PostHog + GenOps")
    print("=" * 55)
    print()

    # Import and setup GenOps PostHog adapter
    try:
        from genops.providers.posthog import GenOpsPostHogAdapter
        print("✅ GenOps PostHog integration loaded")
    except ImportError as e:
        print(f"❌ Failed to import GenOps PostHog: {e}")
        print("💡 Fix: pip install genops[posthog]")
        return False

    # Initialize e-commerce analytics adapter
    print("\n🎯 Setting up E-Commerce Analytics Configuration...")
    adapter = GenOpsPostHogAdapter(
        team="ecommerce-analytics",
        project="online-store-tracking",
        environment="production",
        customer_id="store_main",
        cost_center="marketing",
        daily_budget_limit=150.0,  # Higher budget for e-commerce volume
        governance_policy="advisory",  # Flexible for high-traffic events
        tags={
            'store_type': 'fashion_retail',
            'analytics_tier': 'premium',
            'traffic_volume': 'high',
            'conversion_tracking': 'enabled'
        }
    )

    print("✅ E-commerce adapter configured")
    print(f"   📊 Daily budget: ${adapter.daily_budget_limit}")
    print("   🏪 Store type: fashion retail")
    print("   📈 Expected volume: 50k+ events/day")

    # Simulate complete e-commerce user journey
    print("\n" + "="*55)
    print("🛍️ Simulating Complete E-Commerce User Journey")
    print("="*55)

    # Customer segments for realistic simulation
    customer_segments = [
        {'segment': 'new_visitor', 'conversion_rate': 0.02, 'avg_order_value': 85.00},
        {'segment': 'returning_customer', 'conversion_rate': 0.08, 'avg_order_value': 125.00},
        {'segment': 'vip_customer', 'conversion_rate': 0.15, 'avg_order_value': 350.00},
        {'segment': 'mobile_user', 'conversion_rate': 0.04, 'avg_order_value': 75.00}
    ]

    total_revenue = 0.0
    total_conversions = 0
    total_events = 0

    # Track multiple customer journeys
    for journey_id in range(1, 6):  # 5 customer journeys
        segment = random.choice(customer_segments)
        customer_id = f"customer_{journey_id:03d}"

        print(f"\n🧑‍💼 Customer Journey #{journey_id}: {segment['segment'].replace('_', ' ').title()}")
        print("-" * 50)

        with adapter.track_analytics_session(
            session_name=f"ecommerce_journey_{journey_id}",
            customer_id=customer_id,
            cost_center="ecommerce_operations",
            segment=segment['segment']
        ) as session:

            journey_revenue = 0.0
            events_in_journey = 0

            # 1. Landing and Browsing Phase
            print("📱 Phase 1: Landing & Product Discovery")

            # Landing page view
            result = adapter.capture_event_with_governance(
                event_name="page_viewed",
                properties={
                    "page_type": "landing",
                    "traffic_source": random.choice(["google", "facebook", "direct", "email"]),
                    "device_type": random.choice(["desktop", "mobile", "tablet"]),
                    "customer_segment": segment['segment']
                },
                distinct_id=customer_id,
                session_id=session.session_id
            )
            events_in_journey += 1
            print(f"   ✅ Landing page view tracked - Cost: ${result['cost']:.6f}")

            # Product category browsing
            categories = ["dresses", "shoes", "accessories", "tops", "bottoms"]
            browsed_categories = random.sample(categories, random.randint(2, 4))

            for category in browsed_categories:
                result = adapter.capture_event_with_governance(
                    event_name="category_viewed",
                    properties={
                        "category": category,
                        "products_shown": random.randint(12, 48),
                        "filter_applied": random.choice([True, False]),
                        "customer_segment": segment['segment']
                    },
                    distinct_id=customer_id,
                    session_id=session.session_id
                )
                events_in_journey += 1
                print(f"   🏷️ Category '{category}' browsed - Cost: ${result['cost']:.6f}")

            # 2. Product Interaction Phase
            print("\n📦 Phase 2: Product Interaction & Consideration")

            # Product detail views
            products_viewed = random.randint(3, 8)
            for i in range(products_viewed):
                product_id = f"prod_{random.randint(1000, 9999)}"
                product_price = round(random.uniform(25.0, 200.0), 2)

                result = adapter.capture_event_with_governance(
                    event_name="product_viewed",
                    properties={
                        "product_id": product_id,
                        "product_name": f"Fashion Item #{product_id}",
                        "price": product_price,
                        "category": random.choice(browsed_categories),
                        "view_duration": random.randint(15, 180),
                        "customer_segment": segment['segment']
                    },
                    distinct_id=customer_id,
                    is_identified=True,  # Product views are identified events
                    session_id=session.session_id
                )
                events_in_journey += 1
                print(f"   👀 Product {product_id} viewed (${product_price}) - Cost: ${result['cost']:.6f}")

            # Search behavior
            if random.random() < 0.6:  # 60% of users search
                search_terms = ["red dress", "summer shoes", "evening wear", "casual top"]
                search_term = random.choice(search_terms)

                result = adapter.capture_event_with_governance(
                    event_name="search_performed",
                    properties={
                        "search_query": search_term,
                        "results_count": random.randint(5, 50),
                        "search_type": "product_search",
                        "customer_segment": segment['segment']
                    },
                    distinct_id=customer_id,
                    session_id=session.session_id
                )
                events_in_journey += 1
                print(f"   🔍 Search '{search_term}' performed - Cost: ${result['cost']:.6f}")

            # 3. Shopping Cart Phase
            print("\n🛒 Phase 3: Shopping Cart & Checkout Consideration")

            # Add to cart (based on conversion rate)
            if random.random() < segment['conversion_rate'] * 3:  # Higher add-to-cart rate
                cart_items = random.randint(1, 4)
                cart_total = 0.0

                for item_num in range(cart_items):
                    item_price = round(random.uniform(30.0, segment['avg_order_value']), 2)
                    cart_total += item_price

                    result = adapter.capture_event_with_governance(
                        event_name="add_to_cart",
                        properties={
                            "product_id": f"cart_item_{item_num + 1}",
                            "price": item_price,
                            "quantity": 1,
                            "cart_total": cart_total,
                            "customer_segment": segment['segment']
                        },
                        distinct_id=customer_id,
                        is_identified=True,
                        session_id=session.session_id
                    )
                    events_in_journey += 1
                    print(f"   ➕ Added ${item_price} item to cart - Cost: ${result['cost']:.6f}")

                # Cart abandonment or checkout
                if random.random() < segment['conversion_rate']:
                    # Successful checkout
                    print("\n💳 Phase 4: Successful Checkout & Conversion")

                    # Checkout started
                    result = adapter.capture_event_with_governance(
                        event_name="checkout_started",
                        properties={
                            "cart_value": cart_total,
                            "items_count": cart_items,
                            "checkout_type": "standard",
                            "customer_segment": segment['segment']
                        },
                        distinct_id=customer_id,
                        is_identified=True,
                        session_id=session.session_id
                    )
                    events_in_journey += 1
                    print(f"   🎯 Checkout started (${cart_total:.2f}) - Cost: ${result['cost']:.6f}")

                    # Purchase completed
                    order_id = f"order_{random.randint(10000, 99999)}"
                    result = adapter.capture_event_with_governance(
                        event_name="purchase_completed",
                        properties={
                            "order_id": order_id,
                            "revenue": cart_total,
                            "items_purchased": cart_items,
                            "payment_method": random.choice(["credit_card", "paypal", "apple_pay"]),
                            "shipping_method": random.choice(["standard", "express", "overnight"]),
                            "customer_segment": segment['segment'],
                            "first_purchase": segment['segment'] == 'new_visitor'
                        },
                        distinct_id=customer_id,
                        is_identified=True,
                        session_id=session.session_id
                    )
                    events_in_journey += 1
                    journey_revenue = cart_total
                    total_conversions += 1
                    print(f"   🎉 Purchase completed! Revenue: ${cart_total:.2f} - Cost: ${result['cost']:.6f}")

                else:
                    # Cart abandonment
                    result = adapter.capture_event_with_governance(
                        event_name="cart_abandoned",
                        properties={
                            "cart_value": cart_total,
                            "items_count": cart_items,
                            "abandonment_stage": random.choice(["cart_review", "shipping_info", "payment_info"]),
                            "customer_segment": segment['segment']
                        },
                        distinct_id=customer_id,
                        session_id=session.session_id
                    )
                    events_in_journey += 1
                    print(f"   😞 Cart abandoned (${cart_total:.2f}) - Cost: ${result['cost']:.6f}")

            # Session summary
            total_revenue += journey_revenue
            total_events += events_in_journey

            print("\n📊 Journey Summary:")
            print(f"   Events tracked: {events_in_journey}")
            print(f"   Revenue generated: ${journey_revenue:.2f}")
            print(f"   Customer segment: {segment['segment'].replace('_', ' ').title()}")

            # Small delay to simulate realistic timing
            time.sleep(0.5)

    # Overall analytics summary
    print("\n" + "="*55)
    print("📈 E-Commerce Analytics Summary")
    print("="*55)

    cost_summary = adapter.get_cost_summary()
    conversion_rate = (total_conversions / 5) * 100  # 5 customer journeys

    print("📊 Business Metrics:")
    print(f"   Total revenue tracked: ${total_revenue:.2f}")
    print(f"   Conversions: {total_conversions}/5 ({conversion_rate:.1f}%)")
    print(f"   Average order value: ${total_revenue/max(total_conversions, 1):.2f}")
    print(f"   Events per customer journey: {total_events/5:.1f}")

    print("\n💰 Cost Intelligence:")
    print(f"   Total analytics cost: ${cost_summary['daily_costs']:.6f}")
    print(f"   Cost per event: ${cost_summary['daily_costs']/total_events:.6f}")
    print(f"   Cost per conversion: ${cost_summary['daily_costs']/max(total_conversions, 1):.6f}")
    print(f"   Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")

    print("\n🏛️ Governance Summary:")
    print(f"   Team: {cost_summary['team']}")
    print(f"   Project: {cost_summary['project']}")
    print(f"   Environment: {cost_summary['environment']}")
    print(f"   Policy: {cost_summary['governance_policy']}")
    print(f"   Cost tracking: {'Enabled' if cost_summary['governance_enabled'] else 'Disabled'}")

    # E-commerce specific insights
    print("\n🎯 E-Commerce Analytics Insights:")
    print(f"   ROI on analytics: {(total_revenue / cost_summary['daily_costs']):.0f}x cost")
    print(f"   Revenue per analytics dollar: ${total_revenue / cost_summary['daily_costs']:.2f}")
    print(f"   Estimated monthly analytics cost: ${cost_summary['daily_costs'] * 30:.2f}")
    print(f"   Projected monthly revenue tracking: ${total_revenue * 30:.2f}")

    print("\n✅ E-commerce analytics tracking completed successfully!")
    return True

def get_product_recommendations():
    """Generate realistic product recommendations for e-commerce analytics."""
    return [
        {
            "category": "Conversion Optimization",
            "recommendation": "Track cart abandonment stages for targeted recovery campaigns",
            "implementation": "Add checkout_step_completed events at each stage",
            "expected_impact": "15-25% improvement in conversion rates"
        },
        {
            "category": "Customer Segmentation",
            "recommendation": "Implement behavioral cohort tracking for personalization",
            "implementation": "Add customer_lifecycle_stage to all events",
            "expected_impact": "20-30% increase in customer lifetime value"
        },
        {
            "category": "Cost Optimization",
            "recommendation": "Implement intelligent event sampling for high-volume periods",
            "implementation": "Sample non-critical events during peak traffic",
            "expected_impact": "40-60% reduction in analytics costs"
        }
    ]

if __name__ == "__main__":
    try:
        success = main()

        if success:
            print("\n💡 E-Commerce Analytics Best Practices:")
            recommendations = get_product_recommendations()
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec['category']}: {rec['recommendation']}")
                print(f"      Implementation: {rec['implementation']}")
                print(f"      Expected Impact: {rec['expected_impact']}")
                print()

        exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n👋 E-commerce analytics demonstration interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Error in e-commerce analytics example: {e}")
        print("🔧 Please check your PostHog configuration and try again")
        exit(1)
